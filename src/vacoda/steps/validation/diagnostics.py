"""
vacoda.steps.validation.diagnostics

Build diagnostics for the canonical bilateral Validation workflow.

The diagnostics are deliberately separated into three concepts:

1. source-reporter diagnostics, which describe exporter/importer quantity
   availability and disagreement but do not define model-training eligibility;
2. training-sample diagnostics, which audit the canonical bilateral training
   population defined by HS × year UV support, positive UV MAD, and normalized
   UV distance;
3. model diagnostics, which summarize expanding-window out-of-fold prediction
   performance.

No reporter-side disagreement criterion is used to define the training sample.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import pandas as pd

from vacoda.config.steps.validation import ValidationConfig
from vacoda.steps.validation.training_sample import training_definition_name
from vacoda.utils.display import get_symbols
from vacoda.utils.sql import sql_literal
from vacoda.pipeline.checkpoint import (
    validate_local_checkpoint,
    file_fingerprint,
    write_local_checkpoint,
    semantic_config_fingerprint,
)

PRE_TRAINING_REFERENCE_WINDOWS = (
    ("0_0", 0, "Same year"),
    ("1_0", 1, "Current + 1 previous year"),
    ("2_0", 2, "Current + 2 previous years"),
)

NORMALIZED_UV_CUTOFFS = (
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
)

SOURCE_DISAGREEMENT_BANDS = (
    ("0-5%", 0.00, 0.05),
    ("5-10%", 0.05, 0.10),
    ("10-25%", 0.10, 0.25),
    ("25-50%", 0.25, 0.50),
    ("50-100%", 0.50, 1.00),
)

DIAGNOSTIC_QUANTILES = (
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


# ======================================================================
# Helpers
# ======================================================================


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
    """Require all columns needed by Validation diagnostics."""

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

    missing = sorted(required_columns - available)

    if missing:
        raise RuntimeError(
            "Validation diagnostics input is missing required column(s): "
            + ", ".join(missing)
        )



def _append_metric(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    criterion: str,
    variable: str | None = None,
    value: float | None = None,
    n_rows: int | None = None,
    total_rows: int | None = None,
    share_of_all: float | None = None,
    share_of_parent: float | None = None,
    year: int | None = None,
    target_cmd: str | None = None,
    quantile: float | None = None,
    candidate: str | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    weight: float | None = None,
    uv_support_threshold: float | None = None,
    uv_distance_threshold: float | None = None,
    disagreement_band: str | None = None,
) -> None:
    """Append one normalized diagnostics row."""

    rows.append(
        {
            "scope": scope,
            "criterion": criterion,
            "candidate": candidate,
            "model_name": model_name,
            "model_type": model_type,
            "weight":weight,
            "disagreement_threshold": None,
            "uv_support_threshold": uv_support_threshold,
            "uv_distance_threshold": uv_distance_threshold,
            "disagreement_band": disagreement_band,
            "uv_distance_band": None,
            "year": year,
            "target_cmd": target_cmd,
            "variable": variable,
            "quantile": quantile,
            "n_rows": n_rows,
            "total_rows": total_rows,
            "share_of_all": share_of_all,
            "share_of_parent": share_of_parent,
            "value": value,
        }
    )


# ======================================================================
# Public step
# ======================================================================


def build_diagnostics(
    *,
    cfg: ValidationConfig,
    input_path: Path,
    structural_audit_path: Path,
    model_predictions_path: Path,
    selected_model_path: Path,
    inference_path: Path,
    inference_audit_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Build post-training Validation diagnostics.

    ``input_path`` is the training-sample artifact.  Reporter-side quantities
    are retained only for source-data diagnostics; the model-training population
    is audited exclusively through ``training_eligible`` and the configured
    HS × year unit-value criteria.
    """

    start = perf_counter()

    input_path = Path(input_path)
    structural_audit_path = Path(structural_audit_path)
    model_predictions_path = Path(model_predictions_path)
    selected_model_path = Path(selected_model_path)
    inference_path = Path(inference_path)
    inference_audit_path = Path(inference_audit_path)


    if not input_path.is_file():
        raise FileNotFoundError(
            f"Validation diagnostics input not found: {input_path}"
        )

    if not structural_audit_path.is_file():
        raise FileNotFoundError(
            "Validation diagnostics structural-audit artifact not found: "
            f"{structural_audit_path}"
        )

    if not model_predictions_path.is_file():
        raise FileNotFoundError(
            "Validation model-prediction artifact not found: "
            f"{model_predictions_path}"
        )

    if not selected_model_path.is_file():
        raise FileNotFoundError(
            "Validation selected-model artifact not found: "
            f"{selected_model_path}"
        )

    if not inference_path.is_file():
        raise FileNotFoundError(
            "Validation inference artifact not found: "
            f"{inference_path}"
        )
    if not inference_audit_path.is_file():
        raise FileNotFoundError(
            "Validation inference-audit artifact not found: "
            f"{inference_audit_path}"
        )
    
    training_definition = training_definition_name(
        cfg=cfg
    )

    base_output_path = cfg.product_path(
        "diagnostics"
    )

    output_path = (
        base_output_path.parent
        / training_definition
        / base_output_path.name
    )

    columns_cfg = cfg.diagnostics.columns
    training_cfg = cfg.model.training_sample

    year_col = str(
        columns_cfg.year
    )

    product_col = str(
        columns_cfg.product
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

    quantity_col = "trade_synced_quantity_kg"

    min_reference_observations = int(
        training_cfg.min_hs_year_reference_observations
    )

    max_normalized_uv_distance_mad = float(
        training_cfg.max_normalized_uv_distance_mad
    )

    reference_year_before = int(
        training_cfg.reference_year_window.before
    )

    reference_year_after = int(
        training_cfg.reference_year_window.after
    )


    ## local checkpoints
    checkpoint_path = (
        output_path.parent
        / "checkpoints"
        / "diagnostics.json"
    )

    config_fp = semantic_config_fingerprint(
        config={
            "columns": {
                "year": year_col,
                "product": product_col,
                "nominal_value": nominal_value_col,
                "constant_value": constant_value_col,
                "export_quantity": export_quantity_col,
                "import_quantity": import_quantity_col,
                "export_estimated": export_estimated_col,
                "import_estimated": import_estimated_col,
                "quantity_relative_difference": (
                    relative_difference_col
                ),
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
            "diagnostic_definitions": {
                "reference_windows": list(
                    PRE_TRAINING_REFERENCE_WINDOWS
                ),
                "normalized_uv_cutoffs": list(
                    NORMALIZED_UV_CUTOFFS
                ),
                "source_disagreement_bands": list(
                    SOURCE_DISAGREEMENT_BANDS
                ),
                "quantiles": list(
                    DIAGNOSTIC_QUANTILES
                ),
            },
        }
    )

    input_fp = {
        "training_sample": file_fingerprint(
            input_path,
            hash_file=True,
        ),
        "structural_audit": file_fingerprint(
            structural_audit_path,
            hash_file=True,
        ),
        "model_predictions": file_fingerprint(
            model_predictions_path,
            hash_file=True,
        ),
        "selected_model": file_fingerprint(
            selected_model_path,
            hash_file=True,
        ),
        "inference": file_fingerprint(
            inference_path,
            hash_file=True,
        ),
        "inference_audit": file_fingerprint(
            inference_audit_path,
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
            "%s REUSE — Validation diagnostics "
            "local checkpoint matches: %s",
            get_symbols("reuse"),
            output_path,
        )

        return {
            "status": "reused",
            "run_id": run_id,
            "output_path": output_path,
            "diagnostics_path": output_path,
            "reused": True,
            "fitted": False,
            "selection_already_applied": True,
        }

    logger.info(
        "%s REBUILD — Validation diagnostics: %s",
        get_symbols("rebuilt"),
        "forced" if force else reason,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("  Input: %s", input_path)
    logger.info("  Model predictions: %s", model_predictions_path)
    logger.info("  Output: %s", output_path)


    required_columns = {
        year_col,
        product_col,
        "exporter_iso",
        "importer_iso",
        nominal_value_col,
        constant_value_col,
        export_quantity_col,
        import_quantity_col,
        export_estimated_col,
        import_estimated_col,
        relative_difference_col,
        quantity_col,
        "selected_unit_value",
        "training_eligible",
        "hs_year_uv_reference",
        "hs_year_uv_reference_log",
        "hs_year_uv_reference_n",
        "hs_year_uv_log_mad",
        "selected_uv_log_distance",
        "training_normalized_uv_distance",
        "uv_evidence_available",
        "training_selected",
    }

    con = duckdb.connect()

    try:
        con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")

        # --------------------------------------------------------------
        # Require passing quantity-method audit
        # --------------------------------------------------------------

        _require_columns(
            con=con,
            input_path=input_path,
            required_columns=required_columns,
        )

        input_sql = sql_literal(input_path)
        predictions_sql = sql_literal(model_predictions_path)
        inference_sql = sql_literal(inference_path)
        selected_model_sql = sql_literal(selected_model_path)
        inference_audit_sql = sql_literal(inference_audit_path)

        diagnostics_rows: list[dict[str, Any]] = []


        # --------------------------------------------------------------
        # Production-model selection
        # --------------------------------------------------------------

        selected_model_rows = con.execute(
            f"""
            SELECT
                model_name,
                model_type,
                selection_metric,
                selection_direction,
                selection_value

            FROM read_parquet(
                {selected_model_sql}
            )
            """
        ).fetchall()

        if len(selected_model_rows) != 1:
            raise RuntimeError(
                "Validation selected-model artifact must contain "
                "exactly one row; "
                f"found {len(selected_model_rows):,}."
            )

        (
            selected_model_name,
            selected_model_type,
            selection_metric,
            selection_direction,
            selection_value,
        ) = selected_model_rows[0]

        if (
            selected_model_name is None
            or selected_model_type is None
            or selection_metric is None
            or selection_direction is None
            or selection_value is None
        ):
            raise RuntimeError(
                "Validation selected-model artifact contains "
                "missing selection fields."
            )

        _append_metric(
            diagnostics_rows,
            scope="production_model_selection",
            criterion=str(selection_metric),
            variable=str(selection_direction),
            value=float(selection_value),
            model_name=str(selected_model_name),
            model_type=str(selected_model_type),
        )

        # --------------------------------------------------------------
        # Require passing inference audit
        # --------------------------------------------------------------

        inference_audit = con.execute(
            f"""
            SELECT
                audit_passed,
                n_duplicate_keys,
                n_prediction_availability_mismatches,
                n_prediction_inverse_transform_mismatches,
                n_residual_formula_mismatches,
                n_unit_value_identity_mismatches

            FROM read_parquet(
                {inference_audit_sql}
            )

            LIMIT 1
            """
        ).fetchone()

        if inference_audit is None:
            raise RuntimeError(
                "Validation inference audit is empty."
            )

        (
            inference_audit_passed,
            n_duplicate_keys,
            n_prediction_availability_mismatches,
            n_prediction_inverse_transform_mismatches,
            n_residual_formula_mismatches,
            n_unit_value_identity_mismatches,
        ) = inference_audit

        if not bool(inference_audit_passed):
            raise RuntimeError(
                "Validation diagnostics require a passing "
                "Validation inference audit."
            )

        for criterion, n_rows in (
            (
                "duplicate_bilateral_keys",
                int(n_duplicate_keys),
            ),
            (
                "prediction_availability_mismatches",
                int(n_prediction_availability_mismatches),
            ),
            (
                "prediction_inverse_transform_mismatches",
                int(n_prediction_inverse_transform_mismatches),
            ),
            (
                "residual_formula_mismatches",
                int(n_residual_formula_mismatches),
            ),
            (
                "unit_value_identity_mismatches",
                int(n_unit_value_identity_mismatches),
            ),
        ):
            _append_metric(
                diagnostics_rows,
                scope="inference_audit",
                criterion=criterion,
                n_rows=n_rows,
            )

        _append_metric(
            diagnostics_rows,
            scope="inference_audit",
            criterion="audit_passed",
            value=1.0,
        )
        # --------------------------------------------------------------
        # Inference View
        # --------------------------------------------------------------
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW validation_inference AS

            SELECT
                year,
                exporter_iso,
                importer_iso,
                target_cmd,
                target_hs4,

                trade_synced_quantity_kg,
                training_selected,

                model_input_valid,
                model_prediction_available,

                model_prediction_log_quantity,
                model_prediction_quantity_kg,

                observed_quantity_available,
                observed_quantity_valid_for_log,

                quantity_log_residual,
                quantity_abs_log_residual,

                exporter_seen,
                importer_seen,
                hs6_seen,
                hs4_seen,

                validation_model_name,
                validation_model_type

            FROM read_parquet(
                {inference_sql}
            )
            """
        )

        inference_counts = con.execute(
            """
            SELECT
                COUNT(*) AS n_all,

                COUNT(*) FILTER (
                    WHERE model_input_valid
                ) AS n_model_input_valid,

                COUNT(*) FILTER (
                    WHERE model_prediction_available
                ) AS n_prediction_available,

                COUNT(*) FILTER (
                    WHERE observed_quantity_available
                ) AS n_observed_quantity,

                COUNT(*) FILTER (
                    WHERE NOT observed_quantity_available
                ) AS n_missing_quantity,

                COUNT(*) FILTER (
                    WHERE observed_quantity_valid_for_log
                    AND model_prediction_available
                ) AS n_residual_evaluable

            FROM validation_inference
            """
        ).fetchone()

        (
            n_inference_all,
            n_model_input_valid,
            n_prediction_available,
            n_observed_quantity,
            n_missing_quantity,
            n_residual_evaluable,
        ) = map(int, inference_counts)


        for criterion, n_rows in (
            ("all_rows", n_inference_all),
            ("model_input_valid", n_model_input_valid),
            ("prediction_available", n_prediction_available),
            ("observed_quantity_available", n_observed_quantity),
            ("missing_quantity", n_missing_quantity),
            ("residual_evaluable", n_residual_evaluable),
        ):

            _append_metric(
                diagnostics_rows,
                scope="inference_population",
                criterion=criterion,
                n_rows=n_rows,
                total_rows=n_inference_all,
                share_of_all=_share(
                    n_rows,
                    n_inference_all,
                ),
            )

        support_states = con.execute(
            """
            SELECT
                hs6_seen,
                hs4_seen,
                COUNT(*) AS n_rows

            FROM validation_inference

            WHERE model_prediction_available
            AND hs6_seen IS NOT NULL
            AND hs4_seen IS NOT NULL

            GROUP BY
                hs6_seen,
                hs4_seen

            ORDER BY
                hs6_seen DESC,
                hs4_seen DESC
            """
        ).fetchdf()

        for row in support_states.itertuples(index=False):

            if bool(row.hs6_seen) and bool(row.hs4_seen):
                criterion = "hs6_seen_hs4_seen"

            elif not bool(row.hs6_seen) and bool(row.hs4_seen):
                criterion = "hs6_unseen_hs4_seen"

            elif not bool(row.hs6_seen) and not bool(row.hs4_seen):
                criterion = "hs6_unseen_hs4_unseen"

            else:
                criterion = "hs6_seen_hs4_unseen"

            _append_metric(
                diagnostics_rows,
                scope="inference_support",
                criterion=criterion,
                n_rows=int(row.n_rows),
                total_rows=n_prediction_available,
                share_of_all=_share(
                    int(row.n_rows),
                    n_inference_all,
                ),
                share_of_parent=_share(
                    int(row.n_rows),
                    n_prediction_available,
                ),
            )

        for variable in (
            "exporter_seen",
            "importer_seen",
        ):

            counts = con.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (
                        WHERE {variable} = TRUE
                    ) AS seen_rows,

                    COUNT(*) FILTER (
                        WHERE {variable} = FALSE
                    ) AS unseen_rows

                FROM validation_inference

                WHERE model_prediction_available
                AND {variable} IS NOT NULL
                """
            ).fetchone()

            seen_rows = int(counts[0])
            unseen_rows = int(counts[1])
            evaluated_rows = seen_rows + unseen_rows

            for criterion, n_rows in (
                ("seen", seen_rows),
                ("unseen", unseen_rows),
            ):

                _append_metric(
                    diagnostics_rows,
                    scope="inference_categorical_support",
                    criterion=criterion,
                    variable=variable,
                    n_rows=n_rows,
                    total_rows=evaluated_rows,
                    share_of_parent=_share(
                        n_rows,
                        evaluated_rows,
                    ),
                )

        for quantile in DIAGNOSTIC_QUANTILES:

            value = con.execute(
                f"""
                SELECT
                    QUANTILE_CONT(
                        quantity_abs_log_residual,
                        {float(quantile):.17g}
                    )

                FROM validation_inference

                WHERE observed_quantity_valid_for_log
                AND model_prediction_available
                AND quantity_abs_log_residual IS NOT NULL
                AND isfinite(
                    quantity_abs_log_residual
                )
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="inference_residual_distribution",
                criterion="observed_quantity_vs_model",
                variable="absolute_log_residual",
                quantile=float(quantile),
                value=(
                    float(value)
                    if value is not None
                    else None
                ),
                n_rows=n_residual_evaluable,
            )

        for quantile in DIAGNOSTIC_QUANTILES:

            value = con.execute(
                f"""
                SELECT
                    QUANTILE_CONT(
                        quantity_log_residual,
                        {float(quantile):.17g}
                    )

                FROM validation_inference

                WHERE observed_quantity_valid_for_log
                AND model_prediction_available
                AND quantity_log_residual IS NOT NULL
                AND isfinite(
                    quantity_log_residual
                )
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="inference_residual_distribution",
                criterion="observed_quantity_vs_model",
                variable="signed_log_residual",
                quantile=float(quantile),
                value=(
                    float(value)
                    if value is not None
                    else None
                ),
                n_rows=n_residual_evaluable,
            )

        residual_by_year = con.execute(
            """
            SELECT
                year,
                COUNT(*) AS n_rows,
                AVG(
                    ABS(quantity_log_residual)
                ) AS mean_absolute_log_residual,
                MEDIAN(
                    ABS(quantity_log_residual)
                ) AS median_absolute_log_residual

            FROM validation_inference

            WHERE observed_quantity_valid_for_log
            AND model_prediction_available
            AND quantity_log_residual IS NOT NULL
            AND isfinite(
                quantity_log_residual
            )

            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()


        for row in residual_by_year.itertuples(index=False):

            _append_metric(
                diagnostics_rows,
                scope="inference_residual_by_year",
                criterion="observed_quantity_vs_model",
                variable="mean_absolute_log_residual",
                year=int(row.year),
                n_rows=int(row.n_rows),
                value=float(
                    row.mean_absolute_log_residual
                ),
            )

            _append_metric(
                diagnostics_rows,
                scope="inference_residual_by_year",
                criterion="observed_quantity_vs_model",
                variable="median_absolute_log_residual",
                year=int(row.year),
                n_rows=int(row.n_rows),
                value=float(
                    row.median_absolute_log_residual
                ),
            )

        unseen_products = con.execute(
            """
            SELECT
                target_cmd,
                target_hs4,
                COUNT(*) AS n_rows,
                MIN(year) AS first_year,
                MAX(year) AS last_year,
                COUNT(DISTINCT year) AS n_years

            FROM validation_inference

            WHERE model_prediction_available
            AND hs6_seen = FALSE

            GROUP BY
                target_cmd,
                target_hs4

            ORDER BY
                n_rows DESC,
                target_cmd
            """
        ).fetchdf()

        for row in unseen_products.itertuples(index=False):

            diagnostics_rows.append(
                {
                    "scope": "inference_unseen_product",
                    "criterion": (
                        "hs4_seen"
                        if bool(
                            con.execute(
                                """
                                SELECT BOOL_OR(hs4_seen)
                                FROM validation_inference
                                WHERE target_cmd = ?
                                AND model_prediction_available
                                """,
                                [str(row.target_cmd)],
                            ).fetchone()[0]
                        )
                        else "hs4_unseen"
                    ),
                    "candidate": None,
                    "model_name": None,
                    "weight": None,
                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,
                    "disagreement_band": None,
                    "uv_distance_band": None,
                    "year": None,
                    "target_cmd": str(
                        row.target_cmd
                    ),
                    "variable": "inference_rows",
                    "quantile": None,
                    "n_rows": int(
                        row.n_rows
                    ),
                    "total_rows": n_prediction_available,
                    "share_of_all": _share(
                        int(row.n_rows),
                        n_inference_all,
                    ),
                    "share_of_parent": _share(
                        int(row.n_rows),
                        n_prediction_available,
                    ),
                    "value": None,
                    "target_hs4": str(
                        row.target_hs4
                    ),
                    "first_year": int(
                        row.first_year
                    ),
                    "last_year": int(
                        row.last_year
                    ),
                    "n_years": int(
                        row.n_years
                    ),
                }
            )
        # --------------------------------------------------------------
        # Canonical diagnostics view
        # --------------------------------------------------------------
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW validation_diagnostics_base AS
            SELECT
                "{year_col}" AS year,
                exporter_iso,
                importer_iso,
                "{product_col}" AS target_cmd,

                "{quantity_col}" AS trade_synced_quantity_kg,

                "{nominal_value_col}" AS trade_value_reconciled,
                "{constant_value_col}" AS trade_value_reconciled_constant_usd,

                "{export_quantity_col}" AS export_quantity_kg,
                "{import_quantity_col}" AS import_quantity_kg,
                "{export_estimated_col}" AS export_quantity_estimated,
                "{import_estimated_col}" AS import_quantity_estimated,
                "{relative_difference_col}" AS quantity_relative_difference,

                selected_unit_value,
                training_eligible,
                hs_year_uv_reference,
                hs_year_uv_reference_log,
                hs_year_uv_reference_n,
                hs_year_uv_log_mad,
                selected_uv_log_distance,
                training_normalized_uv_distance,
                uv_evidence_available,
                training_selected,

                (
                    "{export_quantity_col}" IS NOT NULL
                    AND "{export_quantity_col}" > 0
                    AND isfinite(CAST("{export_quantity_col}" AS DOUBLE))
                ) AS has_export_quantity,

                (
                    "{import_quantity_col}" IS NOT NULL
                    AND "{import_quantity_col}" > 0
                    AND isfinite(CAST("{import_quantity_col}" AS DOUBLE))
                ) AS has_import_quantity,

                (
                    "{export_quantity_col}" IS NOT NULL
                    AND "{import_quantity_col}" IS NOT NULL
                    AND "{export_quantity_col}" > 0
                    AND "{import_quantity_col}" > 0
                    AND isfinite(CAST("{export_quantity_col}" AS DOUBLE))
                    AND isfinite(CAST("{import_quantity_col}" AS DOUBLE))
                ) AS has_both_quantities,

                (
                    "{export_quantity_col}" IS NOT NULL
                    AND "{import_quantity_col}" IS NOT NULL
                    AND "{export_quantity_col}" > 0
                    AND "{import_quantity_col}" > 0
                    AND isfinite(CAST("{export_quantity_col}" AS DOUBLE))
                    AND isfinite(CAST("{import_quantity_col}" AS DOUBLE))
                    AND COALESCE("{export_estimated_col}", FALSE) = FALSE
                    AND COALESCE("{import_estimated_col}", FALSE) = FALSE
                ) AS both_quantities_non_estimated

            FROM read_parquet({input_sql})
            """
        )

        # --------------------------------------------------------------
        # Model predictions
        # --------------------------------------------------------------
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW validation_model_predictions AS
            SELECT
                year,
                exporter_iso,
                importer_iso,
                target_cmd,

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
                {predictions_sql}
            )

            WHERE target_transformed IS NOT NULL
            AND prediction_transformed IS NOT NULL
            AND isfinite(target_transformed)
            AND isfinite(prediction_transformed)
            """
        )

        n_all_rows = int(
            con.execute(
                "SELECT COUNT(*) FROM validation_diagnostics_base"
            ).fetchone()[0]
        )

        _append_metric(
            diagnostics_rows,
            scope="quantity_methodology",
            criterion="canonical_bilateral_quantity",
            candidate="value_weighted",
            n_rows=n_all_rows,
            total_rows=n_all_rows,
            share_of_all=1.0,
        )


        for quantile in DIAGNOSTIC_QUANTILES:

            value = con.execute(
                f"""
                SELECT QUANTILE_CONT(
                    selected_uv_log_distance,
                    {float(quantile):.17g}
                )
                FROM validation_diagnostics_base
                WHERE training_eligible
                AND selected_uv_log_distance IS NOT NULL
                AND isfinite(selected_uv_log_distance)
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="training_uv_distance_distribution",
                criterion="canonical_eligible_population",
                variable="selected_uv_log_distance",
                quantile=float(quantile),
                value=(
                    float(value)
                    if value is not None
                    else None
                ),
            )
        # ==================================================================
        # 2. Source-reporter diagnostics
        # ==================================================================
        # These describe the original reporter evidence only.  They are not
        # training-sample criteria and are not used as truth labels.

        source_counts = con.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE has_export_quantity AND has_import_quantity
                ) AS n_both,
                COUNT(*) FILTER (
                    WHERE has_export_quantity AND NOT has_import_quantity
                ) AS n_export_only,
                COUNT(*) FILTER (
                    WHERE NOT has_export_quantity AND has_import_quantity
                ) AS n_import_only,
                COUNT(*) FILTER (
                    WHERE NOT has_export_quantity AND NOT has_import_quantity
                ) AS n_neither,
                COUNT(*) FILTER (
                    WHERE both_quantities_non_estimated
                ) AS n_both_non_estimated
            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_both_quantities,
            n_export_only,
            n_import_only,
            n_neither,
            n_both_non_estimated,
        ) = map(int, source_counts)

        for criterion, n_rows in (
            ("both", n_both_quantities),
            ("export_only", n_export_only),
            ("import_only", n_import_only),
            ("neither", n_neither),
            ("both_non_estimated", n_both_non_estimated),
        ):
            _append_metric(
                diagnostics_rows,
                scope="source_quantity_availability",
                criterion=criterion,
                n_rows=n_rows,
                total_rows=n_all_rows,
                share_of_all=_share(n_rows, n_all_rows),
            )

        n_disagreement_evaluable = int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM validation_diagnostics_base
                WHERE has_both_quantities
                  AND quantity_relative_difference IS NOT NULL
                  AND isfinite(quantity_relative_difference)
                """
            ).fetchone()[0]
        )

        for quantile in DIAGNOSTIC_QUANTILES:
            value = con.execute(
                f"""
                SELECT QUANTILE_CONT(
                    quantity_relative_difference,
                    {float(quantile):.17g}
                )
                FROM validation_diagnostics_base
                WHERE has_both_quantities
                  AND quantity_relative_difference IS NOT NULL
                  AND isfinite(quantity_relative_difference)
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="source_quantity_disagreement_distribution",
                criterion="both_source_quantities_available",
                variable="quantity_relative_difference",
                quantile=float(quantile),
                value=(float(value) if value is not None else None),
                n_rows=n_disagreement_evaluable,
            )

        for index, (label, lower, upper) in enumerate(
            SOURCE_DISAGREEMENT_BANDS
        ):
            lower_condition = (
                f"quantity_relative_difference >= {lower:.17g}"
                if index == 0
                else f"quantity_relative_difference > {lower:.17g}"
            )

            n_rows = int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM validation_diagnostics_base
                    WHERE has_both_quantities
                      AND quantity_relative_difference IS NOT NULL
                      AND isfinite(quantity_relative_difference)
                      AND {lower_condition}
                      AND quantity_relative_difference <= {upper:.17g}
                    """
                ).fetchone()[0]
            )

            _append_metric(
                diagnostics_rows,
                scope="source_quantity_disagreement_band",
                criterion="reporter_disagreement",
                disagreement_band=label,
                n_rows=n_rows,
                total_rows=n_disagreement_evaluable,
                share_of_parent=_share(n_rows, n_disagreement_evaluable),
            )

        # ==================================================================
        # 3. Training-population audit
        # ==================================================================
        # Canonical sequence:
        #   training_eligible
        #   -> support >= configured minimum
        #   -> positive finite MAD
        #   -> UV distance available
        #   -> normalized UV distance <= configured maximum

        funnel = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_all,

                COUNT(*) FILTER (
                    WHERE training_eligible
                ) AS n_eligible,

                COUNT(*) FILTER (
                    WHERE training_eligible
                      AND hs_year_uv_reference_n IS NOT NULL
                      AND hs_year_uv_reference_n >= {min_reference_observations}
                ) AS n_support,

                COUNT(*) FILTER (
                    WHERE training_eligible
                      AND hs_year_uv_reference_n IS NOT NULL
                      AND hs_year_uv_reference_n >= {min_reference_observations}
                      AND hs_year_uv_log_mad IS NOT NULL
                      AND hs_year_uv_log_mad > 0
                      AND isfinite(hs_year_uv_log_mad)
                ) AS n_positive_mad,

                COUNT(*) FILTER (
                    WHERE training_eligible
                      AND hs_year_uv_reference_n IS NOT NULL
                      AND hs_year_uv_reference_n >= {min_reference_observations}
                      AND hs_year_uv_log_mad IS NOT NULL
                      AND hs_year_uv_log_mad > 0
                      AND isfinite(hs_year_uv_log_mad)
                      AND selected_uv_log_distance IS NOT NULL
                      AND isfinite(selected_uv_log_distance)
                ) AS n_uv_evaluable,

                COUNT(*) FILTER (
                    WHERE training_selected
                ) AS n_selected

            FROM validation_diagnostics_base
            """
        ).fetchone()

        (
            n_training_all,
            n_training_eligible,
            n_after_support,
            n_after_positive_mad,
            n_uv_evaluable,
            n_training_selected,
        ) = map(int, funnel)

        funnel_specs = (
            ("all_rows", n_training_all),
            ("training_eligible", n_training_eligible),
            ("support", n_after_support),
            ("positive_uv_mad", n_after_positive_mad),
            ("uv_distance_available", n_uv_evaluable),
            ("training_selected", n_training_selected),
        )

        previous = n_training_all

        for criterion, n_rows in funnel_specs:
            _append_metric(
                diagnostics_rows,
                scope="training_population_funnel",
                criterion=criterion,
                n_rows=n_rows,
                total_rows=previous,
                share_of_all=_share(n_rows, n_training_all),
                share_of_parent=_share(n_rows, previous),
                uv_support_threshold=(
                    float(min_reference_observations)
                    if criterion in {
                        "support",
                        "positive_uv_mad",
                        "uv_distance_available",
                        "training_selected",
                    }
                    else None
                ),
                uv_distance_threshold=(
                    float(max_normalized_uv_distance_mad)
                    if criterion == "training_selected"
                    else None
                ),
            )
            previous = n_rows

        # Guard against an accidental mismatch between the explicit rule above
        # and the upstream training_selected column.
        n_rule_selected = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM validation_diagnostics_base
                WHERE training_eligible
                  AND hs_year_uv_reference_n IS NOT NULL
                  AND hs_year_uv_reference_n >= {min_reference_observations}
                  AND hs_year_uv_log_mad IS NOT NULL
                  AND hs_year_uv_log_mad > 0
                  AND isfinite(hs_year_uv_log_mad)
                  AND selected_uv_log_distance IS NOT NULL
                  AND isfinite(selected_uv_log_distance)
                  AND (
                        selected_uv_log_distance / hs_year_uv_log_mad
                      ) <= {max_normalized_uv_distance_mad:.17g}
                """
            ).fetchone()[0]
        )

        if n_rule_selected != n_training_selected:
            raise RuntimeError(
                "Diagnostics training rule disagrees with training_selected: "
                f"rule={n_rule_selected:,}, flag={n_training_selected:,}."
            )

        _append_metric(
            diagnostics_rows,
            scope="training_definition",
            criterion=training_definition,
            variable="training_selected",
            n_rows=n_training_selected,
            total_rows=n_training_all,
            share_of_all=_share(n_training_selected, n_training_all),
            uv_support_threshold=float(min_reference_observations),
            uv_distance_threshold=float(max_normalized_uv_distance_mad),
        )

        # ==================================================================
        # 4. Training coverage by year and product
        # ==================================================================
        training_by_year = con.execute(
            """
            SELECT
                year,
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (WHERE training_eligible) AS eligible_rows,
                COUNT(*) FILTER (WHERE training_selected) AS selected_rows,
                COUNT(DISTINCT target_cmd) AS total_products,
                COUNT(DISTINCT target_cmd) FILTER (
                    WHERE training_selected
                ) AS selected_products
            FROM validation_diagnostics_base
            GROUP BY year
            ORDER BY year
            """
        ).fetchdf()

        for row in training_by_year.itertuples(index=False):
            _append_metric(
                diagnostics_rows,
                scope="training_coverage_by_year",
                criterion="final_training_sample",
                year=int(row.year),
                variable="training_selected_rows",
                n_rows=int(row.selected_rows),
                total_rows=int(row.total_rows),
                share_of_all=_share(
                    int(row.selected_rows),
                    int(row.total_rows),
                ),
                share_of_parent=_share(
                    int(row.selected_rows),
                    int(row.eligible_rows),
                ),
                value=float(row.selected_products),
            )

        training_by_product = con.execute(
            """
            SELECT
                target_cmd,
                COUNT(*) AS total_rows,
                COUNT(*) FILTER (WHERE training_eligible) AS eligible_rows,
                COUNT(*) FILTER (WHERE training_selected) AS selected_rows,
                COUNT(DISTINCT year) AS validation_years,
                COUNT(DISTINCT year) FILTER (
                    WHERE training_selected
                ) AS training_years,
                MIN(year) FILTER (
                    WHERE training_selected
                ) AS first_training_year,
                MAX(year) FILTER (
                    WHERE training_selected
                ) AS last_training_year
            FROM validation_diagnostics_base
            GROUP BY target_cmd
            ORDER BY selected_rows ASC, target_cmd ASC
            """
        ).fetchdf()

        for row in training_by_product.itertuples(index=False):
            selected_rows = int(row.selected_rows)
            total_rows = int(row.total_rows)
            eligible_rows = int(row.eligible_rows)
            validation_years = int(row.validation_years)
            training_years = int(row.training_years)

            diagnostics_rows.append(
                {
                    "scope": "training_coverage_by_product",
                    "criterion": (
                        "seen" if selected_rows > 0 else "unseen"
                    ),
                    "candidate": None,
                    "model_name": None,
                    "weight":None,
                    "disagreement_threshold": None,
                    "uv_support_threshold": None,
                    "uv_distance_threshold": None,
                    "disagreement_band": None,
                    "uv_distance_band": None,
                    "year": None,
                    "target_cmd": str(row.target_cmd),
                    "variable": "training_selected_rows",
                    "quantile": None,
                    "n_rows": selected_rows,
                    "total_rows": total_rows,
                    "share_of_all": _share(selected_rows, total_rows),
                    "share_of_parent": _share(selected_rows, eligible_rows),
                    "value": None,
                    "eligible_rows": eligible_rows,
                    "validation_years": validation_years,
                    "training_years": training_years,
                    "first_training_year": (
                        int(row.first_training_year)
                        if pd.notna(row.first_training_year)
                        else None
                    ),
                    "last_training_year": (
                        int(row.last_training_year)
                        if pd.notna(row.last_training_year)
                        else None
                    ),
                    "training_year_coverage": _share(
                        training_years,
                        validation_years,
                    ),
                }
            )

        # Cumulative rejection stages for products absent from final training.
        unseen_products = [
            str(row.target_cmd)
            for row in training_by_product.itertuples(index=False)
            if int(row.selected_rows) == 0
        ]

        if unseen_products:
            escaped = ", ".join(
                "'" + value.replace("'", "''") + "'"
                for value in unseen_products
            )

            rejection = con.execute(
                f"""
                SELECT
                    target_cmd,

                    COUNT(*) FILTER (
                        WHERE training_eligible
                    ) AS eligible_rows,

                    COUNT(*) FILTER (
                        WHERE training_eligible
                          AND hs_year_uv_reference_n IS NOT NULL
                          AND hs_year_uv_reference_n >= {min_reference_observations}
                    ) AS after_support_rows,

                    COUNT(*) FILTER (
                        WHERE training_eligible
                          AND hs_year_uv_reference_n IS NOT NULL
                          AND hs_year_uv_reference_n >= {min_reference_observations}
                          AND hs_year_uv_log_mad IS NOT NULL
                          AND hs_year_uv_log_mad > 0
                          AND isfinite(hs_year_uv_log_mad)
                    ) AS after_positive_mad_rows,

                    COUNT(*) FILTER (
                        WHERE training_eligible
                          AND hs_year_uv_reference_n IS NOT NULL
                          AND hs_year_uv_reference_n >= {min_reference_observations}
                          AND hs_year_uv_log_mad IS NOT NULL
                          AND hs_year_uv_log_mad > 0
                          AND isfinite(hs_year_uv_log_mad)
                          AND selected_uv_log_distance IS NOT NULL
                          AND isfinite(selected_uv_log_distance)
                    ) AS uv_evaluable_rows,

                    COUNT(*) FILTER (
                        WHERE training_selected
                    ) AS selected_rows

                FROM validation_diagnostics_base
                WHERE target_cmd IN ({escaped})
                GROUP BY target_cmd
                ORDER BY target_cmd
                """
            ).fetchdf()

            for row in rejection.itertuples(index=False):
                stages = (
                    ("training_eligible", int(row.eligible_rows)),
                    ("reference_support", int(row.after_support_rows)),
                    ("positive_uv_mad", int(row.after_positive_mad_rows)),
                    ("uv_distance_available", int(row.uv_evaluable_rows)),
                    ("normalized_uv_distance", int(row.selected_rows)),
                )

                previous_rows: int | None = None

                for stage, n_rows in stages:
                    _append_metric(
                        diagnostics_rows,
                        scope="training_rejection_by_product",
                        criterion=stage,
                        target_cmd=str(row.target_cmd),
                        variable="cumulative_training_rows",
                        n_rows=n_rows,
                        total_rows=(
                            previous_rows
                            if previous_rows is not None
                            else n_rows
                        ),
                        share_of_parent=(
                            _share(n_rows, previous_rows)
                            if previous_rows is not None
                            else 1.0
                        ),
                        uv_support_threshold=(
                            float(min_reference_observations)
                            if stage != "training_eligible"
                            else None
                        ),
                        uv_distance_threshold=(
                            float(max_normalized_uv_distance_mad)
                            if stage == "normalized_uv_distance"
                            else None
                        ),
                    )
                    previous_rows = n_rows

        # ==================================================================
        # 5. Reference-window diagnostics
        # ==================================================================
        reference_window_queries: list[str] = []

        for window_name, before, _ in PRE_TRAINING_REFERENCE_WINDOWS:
            reference_window_queries.append(
                f"""
                WITH
                focal_groups AS (
                    SELECT DISTINCT
                        year AS focal_year,
                        target_cmd
                    FROM validation_diagnostics_base
                ),
                eligible_reference_rows AS (
                    SELECT
                        year AS source_year,
                        target_cmd,
                        LN(selected_unit_value) AS log_unit_value
                    FROM validation_diagnostics_base
                    WHERE training_eligible
                      AND selected_unit_value IS NOT NULL
                      AND selected_unit_value > 0
                      AND isfinite(selected_unit_value)
                ),
                window_rows AS (
                    SELECT
                        f.focal_year,
                        f.target_cmd,
                        r.log_unit_value
                    FROM focal_groups AS f
                    INNER JOIN eligible_reference_rows AS r
                        ON r.target_cmd = f.target_cmd
                       AND r.source_year BETWEEN
                           f.focal_year - {before}
                           AND f.focal_year
                ),
                centres AS (
                    SELECT
                        focal_year,
                        target_cmd,
                        COUNT(*) AS reference_n,
                        MEDIAN(log_unit_value) AS median_log_unit_value
                    FROM window_rows
                    GROUP BY focal_year, target_cmd
                ),
                deviations AS (
                    SELECT
                        r.focal_year,
                        r.target_cmd,
                        ABS(
                            r.log_unit_value - c.median_log_unit_value
                        ) AS log_absolute_deviation
                    FROM window_rows AS r
                    INNER JOIN centres AS c
                        USING (focal_year, target_cmd)
                ),
                dispersion AS (
                    SELECT
                        focal_year,
                        target_cmd,
                        MEDIAN(log_absolute_deviation) AS log_mad
                    FROM deviations
                    GROUP BY focal_year, target_cmd
                )
                SELECT
                    '{window_name}' AS reference_window,
                    {before} AS reference_before,
                    f.focal_year AS year,
                    f.target_cmd,
                    COALESCE(c.reference_n, 0) AS reference_n,
                    c.median_log_unit_value,
                    d.log_mad
                FROM focal_groups AS f
                LEFT JOIN centres AS c
                    ON c.focal_year = f.focal_year
                   AND c.target_cmd = f.target_cmd
                LEFT JOIN dispersion AS d
                    ON d.focal_year = f.focal_year
                   AND d.target_cmd = f.target_cmd
                """
            )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE pre_training_reference_windows AS
            """
            + "\nUNION ALL\n".join(
                f"SELECT * FROM ({query})"
                for query in reference_window_queries
            )
        )

        for window_name, before, _ in PRE_TRAINING_REFERENCE_WINDOWS:
            row = con.execute(
                """
                SELECT
                    COUNT(*) AS n_groups,
                    COUNT(*) FILTER (
                        WHERE reference_n > 0
                    ) AS n_with_reference,
                    COUNT(*) FILTER (
                        WHERE reference_n >= ?
                    ) AS n_with_support,
                    MEDIAN(reference_n) AS median_support,
                    MEDIAN(log_mad) FILTER (
                        WHERE log_mad IS NOT NULL
                    ) AS median_log_mad
                FROM pre_training_reference_windows
                WHERE reference_window = ?
                """,
                [min_reference_observations, window_name],
            ).fetchone()

            (
                n_groups,
                n_with_reference,
                n_with_support,
                median_support,
                median_log_mad,
            ) = row

            _append_metric(
                diagnostics_rows,
                scope="reference_window_summary",
                criterion=window_name,
                variable="groups_with_support",
                n_rows=int(n_with_support),
                total_rows=int(n_groups),
                share_of_parent=_share(int(n_with_support), int(n_groups)),
                value=(
                    float(median_support)
                    if median_support is not None
                    else None
                ),
                uv_support_threshold=float(min_reference_observations),
            )

            _append_metric(
                diagnostics_rows,
                scope="reference_window_summary",
                criterion=window_name,
                variable="median_log_mad",
                n_rows=int(n_with_reference),
                total_rows=int(n_groups),
                value=(
                    float(median_log_mad)
                    if median_log_mad is not None
                    else None
                ),
            )

        # ==================================================================
        # 6. Current-reference distributions
        # ==================================================================
        for quantile in DIAGNOSTIC_QUANTILES:
            support_value = con.execute(
                f"""
                SELECT QUANTILE_CONT(
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

            _append_metric(
                diagnostics_rows,
                scope="training_reference_support_distribution",
                criterion="hs_year_groups",
                variable="hs_year_uv_reference_n",
                quantile=float(quantile),
                value=(
                    float(support_value)
                    if support_value is not None
                    else None
                ),
            )

            mad_value = con.execute(
                f"""
                SELECT QUANTILE_CONT(
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
                      AND isfinite(hs_year_uv_log_mad)
                )
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="training_uv_mad_distribution",
                criterion="hs_year_groups",
                variable="hs_year_uv_log_mad",
                quantile=float(quantile),
                value=(float(mad_value) if mad_value is not None else None),
            )

            normalized_value = con.execute(
                f"""
                SELECT QUANTILE_CONT(
                    training_normalized_uv_distance,
                    {float(quantile):.17g}
                )
                FROM validation_diagnostics_base
                WHERE training_eligible
                  AND hs_year_uv_reference_n IS NOT NULL
                  AND hs_year_uv_reference_n >= {min_reference_observations}
                  AND hs_year_uv_log_mad IS NOT NULL
                  AND hs_year_uv_log_mad > 0
                  AND isfinite(hs_year_uv_log_mad)
                  AND training_normalized_uv_distance IS NOT NULL
                  AND isfinite(training_normalized_uv_distance)
                """
            ).fetchone()[0]

            _append_metric(
                diagnostics_rows,
                scope="training_normalized_uv_distance_distribution",
                criterion="pre_cutoff_population",
                variable="training_normalized_uv_distance",
                quantile=float(quantile),
                value=(
                    float(normalized_value)
                    if normalized_value is not None
                    else None
                ),
                n_rows=n_uv_evaluable,
            )

        # ==================================================================
        # 7. Normalized-UV cutoff sensitivity
        # ==================================================================
        for cutoff in NORMALIZED_UV_CUTOFFS:
            n_rows = int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM validation_diagnostics_base
                    WHERE training_eligible
                      AND hs_year_uv_reference_n IS NOT NULL
                      AND hs_year_uv_reference_n >= {min_reference_observations}
                      AND hs_year_uv_log_mad IS NOT NULL
                      AND hs_year_uv_log_mad > 0
                      AND isfinite(hs_year_uv_log_mad)
                      AND training_normalized_uv_distance IS NOT NULL
                      AND isfinite(training_normalized_uv_distance)
                      AND training_normalized_uv_distance <= {float(cutoff):.17g}
                    """
                ).fetchone()[0]
            )

            _append_metric(
                diagnostics_rows,
                scope="training_normalized_uv_cutoff_sensitivity",
                criterion="canonical_pre_cutoff_population",
                variable="training_normalized_uv_distance",
                n_rows=n_rows,
                total_rows=n_uv_evaluable,
                share_of_all=_share(n_rows, n_training_all),
                share_of_parent=_share(n_rows, n_uv_evaluable),
                uv_support_threshold=float(min_reference_observations),
                uv_distance_threshold=float(cutoff),
                value=float(cutoff),
            )

        # ==================================================================
        # 8. Model evaluation
        # ==================================================================
        overall_models = con.execute(
            """
            SELECT
                model_name,
                COUNT(*) AS n_rows,
                AVG(absolute_log_error) AS mae_log,
                SQRT(AVG(squared_log_error)) AS rmse_log,
                MEDIAN(absolute_log_error) AS median_absolute_log_error
            FROM validation_model_predictions
            GROUP BY model_name
            ORDER BY model_name
            """
        ).fetchdf()

        for row in overall_models.itertuples(index=False):
            for variable, value in (
                ("mae_log", row.mae_log),
                ("rmse_log", row.rmse_log),
                (
                    "median_absolute_log_error",
                    row.median_absolute_log_error,
                ),
            ):
                _append_metric(
                    diagnostics_rows,
                    scope="model_performance_overall",
                    criterion="out_of_fold",
                    model_name=str(row.model_name),
                    variable=variable,
                    value=float(value),
                    n_rows=int(row.n_rows),
                )

        by_year_models = con.execute(
            """
            SELECT
                model_name,
                held_out_year,
                COUNT(*) AS n_rows,
                AVG(absolute_log_error) AS mae_log,
                SQRT(AVG(squared_log_error)) AS rmse_log,
                MEDIAN(absolute_log_error) AS median_absolute_log_error
            FROM validation_model_predictions
            GROUP BY model_name, held_out_year
            ORDER BY model_name, held_out_year
            """
        ).fetchdf()

        for row in by_year_models.itertuples(index=False):
            for variable, value in (
                ("mae_log", row.mae_log),
                ("rmse_log", row.rmse_log),
                (
                    "median_absolute_log_error",
                    row.median_absolute_log_error,
                ),
            ):
                _append_metric(
                    diagnostics_rows,
                    scope="model_performance_by_year",
                    criterion="out_of_fold",
                    model_name=str(row.model_name),
                    year=int(row.held_out_year),
                    variable=variable,
                    value=float(value),
                    n_rows=int(row.n_rows),
                )


        # ==================================================================
        # Model ensemble-pair diagnostics
        # ==================================================================
        #
        # Compare signed out-of-fold log errors for the same canonical
        # validation rows.
        #
        # For each model pair, retain:
        #   - residual correlation
        #   - MAE of each model on the paired rows
        #   - MAE difference
        #   - mean pair MAE
        #
        # Low residual correlation together with strong individual MAE
        # indicates potentially useful diversity for model ensembling.
        # ==================================================================

        ensemble_pair_diagnostics = con.execute(
            """
            SELECT
                left_model.model_name
                    AS model_name,

                right_model.model_name
                    AS candidate,

                COUNT(*)
                    AS n_rows,

                CORR(
                    left_model.log_error,
                    right_model.log_error
                )
                    AS residual_correlation,

                AVG(
                    left_model.absolute_log_error
                )
                    AS model_mae_log,

                AVG(
                    right_model.absolute_log_error
                )
                    AS candidate_mae_log,

                ABS(
                    AVG(
                        left_model.absolute_log_error
                    )
                    -
                    AVG(
                        right_model.absolute_log_error
                    )
                )
                    AS mae_difference,

                (
                    AVG(
                        left_model.absolute_log_error
                    )
                    +
                    AVG(
                        right_model.absolute_log_error
                    )
                ) / 2.0
                    AS mean_pair_mae_log

            FROM validation_model_predictions
                AS left_model

            INNER JOIN validation_model_predictions
                AS right_model

                ON left_model.year
                    = right_model.year

                AND left_model.exporter_iso
                    = right_model.exporter_iso

                AND left_model.importer_iso
                    = right_model.importer_iso

                AND left_model.target_cmd
                    = right_model.target_cmd

                AND left_model.held_out_year
                    = right_model.held_out_year

                AND left_model.model_name
                    < right_model.model_name

            GROUP BY
                left_model.model_name,
                right_model.model_name

            ORDER BY
                residual_correlation,
                mean_pair_mae_log
            """
        ).fetchdf()

        for row in ensemble_pair_diagnostics.itertuples(index=False):

            _append_metric(
                diagnostics_rows,
                scope=(
                    "model_residual_correlation"
                ),
                criterion=(
                    "paired_out_of_fold_log_errors"
                ),
                model_name=str(
                    row.model_name
                ),
                candidate=str(
                    row.candidate
                ),
                variable=(
                    "residual_correlation"
                ),
                value=float(
                    row.residual_correlation
                ),
                n_rows=int(
                    row.n_rows
                ),
            )

        for row in ensemble_pair_diagnostics.itertuples(index=False):

            for variable, value in (
                (
                    "residual_correlation",
                    row.residual_correlation,
                ),
                (
                    "model_mae_log",
                    row.model_mae_log,
                ),
                (
                    "candidate_mae_log",
                    row.candidate_mae_log,
                ),
                (
                    "mae_difference",
                    row.mae_difference,
                ),
                (
                    "mean_pair_mae_log",
                    row.mean_pair_mae_log,
                ),
            ):

                _append_metric(
                    diagnostics_rows,
                    scope=(
                        "model_ensemble_pair_diagnostic"
                    ),
                    criterion=(
                        "paired_out_of_fold_log_errors"
                    ),
                    model_name=str(
                        row.model_name
                    ),
                    candidate=str(
                        row.candidate
                    ),
                    variable=variable,
                    value=float(
                        value
                    ),
                    n_rows=int(
                        row.n_rows
                    ),
                )
        
        

        # ==================================================================
        # Weighted ensemble diagnostics
        # ==================================================================
        #
        # Evaluate simple weighted blends of two candidate models using
        # out-of-fold predictions on identical held-out canonical rows.
        #
        # blended_prediction
        #   = w * prediction_model_a
        #   + (1 - w) * prediction_model_b
        #
        # Metrics are computed on the transformed (log) target scale.
        # ==================================================================

        ensemble_pairs = [
            (
                "emb_mlp_bs4096_lr0p001",
                "extra_trees",
            ),
            (
                "emb_mlp_bs4096_lr0p001",
                "random_forest",
            ),
            (
                "emb_mlp_bs4096_lr0p001",
                "histogram_gradient_boosting",
            ),
            (
                "emb_mlp_bs4096_lr0p001",
                "ridge",
            ),
        ]

        ensemble_weights = [
            0.00,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ]

        pair_sql = ", ".join(
            (
                f"('{model_a}', '{model_b}')"
            )
            for model_a, model_b in ensemble_pairs
        )

        weight_sql = ", ".join(
            f"({weight:.6f})"
            for weight in ensemble_weights
        )

        weighted_ensemble_curve = con.execute(
            f"""
            WITH pair_config(
                model_a,
                model_b
            ) AS (
                VALUES
                    {pair_sql}
            ),

            weight_grid(
                weight_model_a
            ) AS (
                VALUES
                    {weight_sql}
            ),

            paired_predictions AS (

                SELECT
                    pairs.model_a,
                    pairs.model_b,

                    weights.weight_model_a,

                    left_model.year,
                    left_model.exporter_iso,
                    left_model.importer_iso,
                    left_model.target_cmd,
                    left_model.held_out_year,

                    left_model.target_transformed,

                    left_model.prediction_transformed
                        AS prediction_model_a,

                    right_model.prediction_transformed
                        AS prediction_model_b

                FROM pair_config
                    AS pairs

                INNER JOIN validation_model_predictions
                    AS left_model

                    ON left_model.model_name
                        = pairs.model_a

                INNER JOIN validation_model_predictions
                    AS right_model

                    ON right_model.model_name
                        = pairs.model_b

                    AND left_model.year
                        = right_model.year

                    AND left_model.exporter_iso
                        = right_model.exporter_iso

                    AND left_model.importer_iso
                        = right_model.importer_iso

                    AND left_model.target_cmd
                        = right_model.target_cmd

                    AND left_model.held_out_year
                        = right_model.held_out_year

                CROSS JOIN weight_grid
                    AS weights
            ),

            scored AS (

                SELECT
                    model_a,
                    model_b,
                    weight_model_a,

                    target_transformed,

                    (
                        weight_model_a
                        * prediction_model_a
                        + (1.0 - weight_model_a)
                        * prediction_model_b
                    ) AS blended_prediction

                FROM paired_predictions
            )

            SELECT
                model_a,
                model_b,
                weight_model_a,

                COUNT(*)
                    AS n_rows,

                AVG(
                    ABS(
                        target_transformed
                        - blended_prediction
                    )
                ) AS mae_log,

                MEDIAN(
                    ABS(
                        target_transformed
                        - blended_prediction
                    )
                ) AS median_absolute_log_error,

                SQRT(
                    AVG(
                        POWER(
                            target_transformed
                            - blended_prediction,
                            2
                        )
                    )
                ) AS rmse_log

            FROM scored

            GROUP BY
                model_a,
                model_b,
                weight_model_a

            ORDER BY
                model_a,
                model_b,
                weight_model_a
            """
        ).fetchdf()
        

        for row in weighted_ensemble_curve.itertuples(
            index=False
        ):
            for variable, value in (
                (
                    "mae_log",
                    row.mae_log,
                ),
                (
                    "median_absolute_log_error",
                    row.median_absolute_log_error,
                ),
                (
                    "rmse_log",
                    row.rmse_log,
                ),
            ):
                _append_metric(
                    diagnostics_rows,
                    scope=(
                        "model_weighted_ensemble_curve"
                    ),
                    criterion=(
                        "out_of_fold_weighted_blend"
                    ),
                    model_name=str(
                        row.model_a
                    ),
                    candidate=str(
                        row.model_b
                    ),
                    variable=variable,
                    value=float(
                        value
                    ),
                    n_rows=int(
                        row.n_rows
                    ),
                    weight=float(
                        row.weight_model_a
                    ),
                )

        weighted_ensemble_best = con.execute(
            f"""
            WITH pair_config(
                model_a,
                model_b
            ) AS (
                VALUES
                    {pair_sql}
            ),

            weight_grid(
                weight_model_a
            ) AS (
                VALUES
                    {weight_sql}
            ),

            paired_predictions AS (

                SELECT
                    pairs.model_a,
                    pairs.model_b,

                    weights.weight_model_a,

                    left_model.target_transformed,

                    left_model.prediction_transformed
                        AS prediction_model_a,

                    right_model.prediction_transformed
                        AS prediction_model_b

                FROM pair_config
                    AS pairs

                INNER JOIN validation_model_predictions
                    AS left_model

                    ON left_model.model_name
                        = pairs.model_a

                INNER JOIN validation_model_predictions
                    AS right_model

                    ON right_model.model_name
                        = pairs.model_b

                    AND left_model.year
                        = right_model.year

                    AND left_model.exporter_iso
                        = right_model.exporter_iso

                    AND left_model.importer_iso
                        = right_model.importer_iso

                    AND left_model.target_cmd
                        = right_model.target_cmd

                    AND left_model.held_out_year
                        = right_model.held_out_year

                CROSS JOIN weight_grid
                    AS weights
            ),

            scored AS (

                SELECT
                    model_a,
                    model_b,
                    weight_model_a,

                    AVG(
                        ABS(
                            target_transformed
                            - (
                                weight_model_a
                                * prediction_model_a
                                + (1.0 - weight_model_a)
                                * prediction_model_b
                            )
                        )
                    ) AS mae_log

                FROM paired_predictions

                GROUP BY
                    model_a,
                    model_b,
                    weight_model_a
            ),

            model_quality AS (

                SELECT
                    model_name,
                    AVG(absolute_log_error)
                        AS mae_log
                FROM validation_model_predictions
                GROUP BY
                    model_name
            )

            SELECT
                scored.model_a,
                scored.model_b,
                scored.weight_model_a
                    AS best_weight_model_a,
                scored.mae_log
                    AS best_ensemble_mae_log,

                quality_a.mae_log
                    AS model_a_mae_log,

                quality_b.mae_log
                    AS model_b_mae_log,

                LEAST(
                    quality_a.mae_log,
                    quality_b.mae_log
                ) AS better_single_mae_log,

                LEAST(
                    quality_a.mae_log,
                    quality_b.mae_log
                ) - scored.mae_log
                    AS improvement_vs_best_single

            FROM scored

            INNER JOIN model_quality
                AS quality_a
                ON scored.model_a = quality_a.model_name

            INNER JOIN model_quality
                AS quality_b
                ON scored.model_b = quality_b.model_name

            QUALIFY
                ROW_NUMBER() OVER (
                    PARTITION BY
                        scored.model_a,
                        scored.model_b
                    ORDER BY
                        scored.mae_log,
                        ABS(scored.weight_model_a - 0.5)
                ) = 1

            ORDER BY
                best_ensemble_mae_log
            """
        ).fetchdf()

        for row in weighted_ensemble_best.itertuples(
            index=False
        ):
            for variable, value in (
                (
                    "best_weight_model_a",
                    row.best_weight_model_a,
                ),
                (
                    "best_ensemble_mae_log",
                    row.best_ensemble_mae_log,
                ),
                (
                    "model_a_mae_log",
                    row.model_a_mae_log,
                ),
                (
                    "model_b_mae_log",
                    row.model_b_mae_log,
                ),
                (
                    "better_single_mae_log",
                    row.better_single_mae_log,
                ),
                (
                    "improvement_vs_best_single",
                    row.improvement_vs_best_single,
                ),
            ):
                _append_metric(
                    diagnostics_rows,
                    scope=(
                        "model_weighted_ensemble_best"
                    ),
                    criterion=(
                        "best_out_of_fold_weighted_blend"
                    ),
                    model_name=str(
                        row.model_a
                    ),
                    candidate=str(
                        row.model_b
                    ),
                    variable=variable,
                    value=float(
                        value
                    ),
                    n_rows=None,
                )
        # ==================================================================
        # Diagnostics artifact
        # ==================================================================
        diagnostics = pd.DataFrame(diagnostics_rows)

        ready_for_reporting = (
            n_all_rows > 0
            and len(diagnostics) > 0
            and n_training_eligible > 0
            and n_training_selected > 0
        )

        if n_all_rows <= 0:
            raise RuntimeError(
                "Validation diagnostics contain no Validation rows."
            )

        if len(diagnostics) <= 0:
            raise RuntimeError(
                "Validation diagnostics artifact contains no diagnostics."
            )

        if n_training_eligible <= 0:
            raise RuntimeError(
                "Validation diagnostics contain no "
                "training-eligible observations."
            )

        if n_training_selected <= 0:
            raise RuntimeError(
                "Validation diagnostics contain no selected "
                "training observations."
            )
        
        if diagnostics.empty:
            raise RuntimeError("Validation diagnostics contain no rows.")


        optional_columns = (
            "candidate",
            "model_name",
            "model_type",
            "weight",
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
            "share_of_all",
            "share_of_parent",
            "value",
            "eligible_rows",
            "validation_years",
            "training_years",
            "first_training_year",
            "last_training_year",
            "training_year_coverage",
            "target_hs4",
            "first_year",
            "last_year",
            "n_years",
        )

        for column in optional_columns:
            if column not in diagnostics.columns:
                diagnostics[column] = None

        diagnostics = diagnostics[
            [
                "scope",
                "criterion",
                "candidate",
                "model_name",
                "model_type",
                "weight",
                "disagreement_threshold",
                "uv_support_threshold",
                "uv_distance_threshold",
                "disagreement_band",
                "uv_distance_band",
                "year",
                "target_cmd",
                "target_hs4",
                "variable",
                "quantile",
                "n_rows",
                "total_rows",
                "share_of_all",
                "share_of_parent",
                "value",
                "eligible_rows",
                "validation_years",
                "training_years",
                "first_training_year",
                "last_training_year",
                "training_year_coverage",
                "first_year",
                "last_year",
                "n_years",
            ]
        ]

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        con.register("_validation_diagnostics", diagnostics)

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM _validation_diagnostics
            )
            TO {sql_literal(temporary_path)}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        temporary_path.replace(output_path)
        
        if ready_for_reporting:
            write_local_checkpoint(
                checkpoint_path=checkpoint_path,
                step_name="diagnostics",
                run_id=run_id,
                config=config_fp,
                input_fingerprint=input_fp,
                output_path=output_path,
            )

    finally:
        con.close()

    elapsed = perf_counter() - start

    logger.info("  All Validation rows: %s", f"{n_all_rows:,}")
    logger.info(
        "  All training-eligible rows: %s",
        f"{n_training_eligible:,}",
    )
    logger.info(
        "  Final selected training rows: %s",
        f"{n_training_selected:,}",
    )

    logger.info("  Diagnostics rows: %s", f"{len(diagnostics):,}")
    logger.info("  Output: %s", output_path)
    logger.info(
        "  Selected model: %s",
        selected_model_path,
    )

    logger.info(
        "  Inference: %s",
        inference_path,
    )

    logger.info(
        "  Inference audit: %s",
        inference_audit_path,
    )
    logger.info(
        "%s CHECK — Validation diagnostics built.",
        get_symbols("check"),
    )
    logger.info("  Ready for reporting: %s", ready_for_reporting)

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "diagnostics_path": output_path,
        "n_rows": n_all_rows,
        "n_training_eligible_rows": n_training_eligible,
        "n_training_selected_rows": n_training_selected,
        "n_both_source_quantities": n_both_quantities,
        "fitted": False,
        "selection_already_applied": True,
        "duration_seconds": elapsed,
        "ready_for_reporting": ready_for_reporting,
        "reused": False,
    }
