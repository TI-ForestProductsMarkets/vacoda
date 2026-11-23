"""
vacoda.steps.validation.workflow

Orchestrate VACODA Validation, including structural audits,
model development, inference, diagnostics, and reporting.

The workflow exposes ready_for_release to the master pipeline.
Publication is handled by the separate Release module.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
from dataclasses import replace

from vacoda.config.steps.validation import ValidationConfig

from vacoda.steps.validation.prepare_input import prepare_validation_input
from vacoda.steps.validation.structural import  audit_structural_integrity
from vacoda.steps.validation.training_sample import build_training_sample
from vacoda.steps.validation.model_training import train_validation_models
from vacoda.steps.validation.model_selection import select_validation_model
from vacoda.steps.validation.model_fitting import fit_selected_validation_model
from vacoda.steps.validation.inference import run_validation_inference
from vacoda.steps.validation.value_model_selection import compare_and_select_value_model
from vacoda.steps.validation.release_gate import evaluate_release_readiness
from vacoda.steps.validation.diagnostics import  build_diagnostics
from vacoda.steps.validation.audit_inference import audit_validation_inference
from vacoda.steps.validation.plausibility import build_plausibility
from vacoda.steps.validation.reporting import build_validation_report
from vacoda.utils import logger
from vacoda.utils.artifacts import (
    require_artifact,
    require_enabled_products,
)
from vacoda.utils.runtime import (
    run_step,
    separator,
    timed_workflow,
)

REQUIRED_PRODUCTS = (
    "prepared_input",
    "structural_audit",
    "training_sample",
    "model_predictions",
    "selected_model",
    "fitted_model",
    "value_selected_model",
    "value_fitted_model",
    "inference",
    "inference_audit",
    "quantity_plausibility",
    "release_readiness",
    "diagnostics",
)

WORKFLOW_STEPS = [
    "Prepare Validation input",
    "Audit Structural integrity",
    "Build Training sample",
    "Train Validation Models",
    "Select Validation Model",
    "Fit Selected Validation Model",
    "Fit Fallback Validation Model",
    "Compare and Select Value Validation Model",
    "Fit Value Validation Model",
    "Run Validation inference",
    "Audit cross-variable consistency",
    "Build Plausibility",
    "Evaluate release readiness",
    "Build Diagnostics",
    "Build Validation decision report",
]


TOTAL_WORKFLOW_STEPS = len(WORKFLOW_STEPS)


def run_validation_workflow(
    *,
    cfg: ValidationConfig,
    run_id: str,
    logger,
    force: bool = False,
    input_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run the complete VACODA Validation workflow.

    Parameters
    ----------
    cfg:
        Parsed Validation configuration.
    run_id:
        Master-pipeline run identifier.
    logger:
        Configured workflow logger.
    force:
        Ignore reusable artifacts inside Validation step modules.
    input_path:
        Optional explicit Value-step output. When omitted,
        ``cfg.input_path()`` is used.

    Expected step modules
    ---------------------
    vacoda.steps.validation.prepare_input
        prepare_validation_input

    vacoda.steps.validation.structural
        audit_structural_integrity

    vacoda.steps.validation.missingness
        build_missingness_diagnostics

    vacoda.steps.validation.unit_values
        build_unit_value_diagnostics

    vacoda.steps.validation.outliers
        detect_validation_outliers

    vacoda.steps.validation.consistency
        audit_cross_variable_consistency

    vacoda.steps.validation.finalize
        build_validated_trade

    vacoda.steps.validation.release
        build_release_dataset

    vacoda.steps.validation.reporting
        build_validation_report
    """

    resolved_input_path = (
        Path(input_path)
        if input_path is not None
        else cfg.input_path()
    )

    if not resolved_input_path.is_file():
        raise FileNotFoundError(
            "Validation input parquet not found: "
            f"{resolved_input_path}"
        )

    enabled_products = cfg.enabled_products()

    if enabled_products["value_fitted_model"] and cfg.value_model is None:
        raise ValueError("Enabled value_fitted_model requires validation.value_model.")

    require_enabled_products(
        workflow_name="Validation",
        enabled_products=enabled_products,
        required_products=REQUIRED_PRODUCTS,
    )

    cfg.output_dir().mkdir(
        parents=True,
        exist_ok=True,
    )

    workflow_start = perf_counter()

    with timed_workflow(logger, "Validation"):

        logger.info("Run ID: %s", run_id)
        logger.info("Target HS: %s", cfg.scope.target_hs)
        logger.info(
            "Year range: %s-%s",
            cfg.scope.min_year,
            cfg.scope.max_year,
        )
        logger.info("Input: %s", resolved_input_path)
        logger.info("Output directory: %s", cfg.output_dir())
        logger.info(
            "Enabled products: %s",
            " ".join(
                f"{name}={enabled}"
                for name, enabled in enabled_products.items()
            ),
        )

        # ==================================================================
        # Step 1 — Prepare Validation input
        # ==================================================================
        prepared_result = run_step(
            logger=logger,
            step_number=1,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[0],
            runner=lambda: prepare_validation_input(
                cfg=cfg,
                input_path=resolved_input_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        prepared_path = require_artifact(
            result=prepared_result,
            path_key="prepared_input_path",
            consumer_step=WORKFLOW_STEPS[1],
            producer_product="prepared Validation input",
        )

        # ==================================================================
        # Step 2 — Structural integrity audit
        # ==================================================================

        structural_result = run_step(
            logger=logger,
            step_number=2,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[1],
            runner=lambda: audit_structural_integrity(
                cfg=cfg,
                input_path=prepared_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        structural_audit_path = require_artifact(
            result=structural_result,
            path_key="structural_audit_path",
            consumer_step=WORKFLOW_STEPS[2],
            producer_product="Validation structural audit",
        )


        # ==================================================================
        # Step 3 — Build training sample
        # ==================================================================

        training_sample_result = run_step(
            logger=logger,
            step_number=3,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[2],
            runner=lambda: build_training_sample(
                cfg=cfg,
                input_path=prepared_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        training_sample_path = require_artifact(
            result=training_sample_result,
            path_key="training_sample_path",
            consumer_step=WORKFLOW_STEPS[3],
            producer_product="training sample",
        )


        # ==================================================================
        # Step 4 — Model Training
        # ==================================================================

        model_predictions_result = run_step(
            logger=logger,
            step_number=4,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[3],
            runner=lambda: train_validation_models(
                cfg=cfg,
                input_path=training_sample_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        model_predictions_path = require_artifact(
            result=model_predictions_result,
            path_key="model_predictions_path",
            consumer_step=WORKFLOW_STEPS[4],
            producer_product=(
                "Validation model predictions"
            ),
        )
        

        # ==================================================================
        # Step 5 — Model Selection
        # ==================================================================

        selected_model_result = run_step(
            logger=logger,
            step_number=5,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[4],
            runner=lambda: select_validation_model(
                cfg=cfg,
                model_predictions_path=model_predictions_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        selected_model_path = require_artifact(
            result=selected_model_result,
            path_key="selected_model_path",
            consumer_step=WORKFLOW_STEPS[5],
            producer_product="selected Validation model",
        )

        # ==================================================================
        # Step 6 — Fit Selected Validation Model
        # ==================================================================

        fitted_model_result = run_step(
            logger=logger,
            step_number=6,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[5],
            runner=lambda: fit_selected_validation_model(
                cfg=cfg,
                training_sample_path=training_sample_path,
                selected_model_path=selected_model_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        fitted_model_path = require_artifact(
            result=fitted_model_result,
            path_key="fitted_model_path",
            consumer_step=WORKFLOW_STEPS[6],
            producer_product="fitted Validation model",
        )

        model_bundle_path = require_artifact(
            result=fitted_model_result,
            path_key="model_bundle_path",
            consumer_step=WORKFLOW_STEPS[6],
            producer_product="fitted Validation model bundle",
        )

        # ==================================================================
        # Step 7 — Fit Fallback Validation Model
        # ==================================================================

        primary_model_type = fitted_model_result["model_type"]
        selected_fallback_model_path = None
        fallback_fitted_model_path = None
        fallback_model_bundle_path = None
        fallback_model_name = None
        fallback_model_type = None

        if primary_model_type == "embedding_mlp":
            selected_fallback_model_path = require_artifact(
                result=selected_model_result,
                path_key="selected_fallback_model_path",
                consumer_step=WORKFLOW_STEPS[6],
                producer_product="selected fallback Validation model",
            )

            fallback_fitted_result = run_step(
                logger=logger,
                step_number=7,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[6],
                runner=lambda: fit_selected_validation_model(
                    cfg=cfg,
                    training_sample_path=training_sample_path,
                    selected_model_path=selected_fallback_model_path,
                    run_id=run_id,
                    role="fallback",
                    force=force,
                    logger=logger,
                ),
            )

            fallback_fitted_model_path = require_artifact(
                result=fallback_fitted_result,
                path_key="fitted_model_path",
                consumer_step=WORKFLOW_STEPS[7],
                producer_product="fitted fallback Validation model",
            )
            fallback_model_bundle_path = require_artifact(
                result=fallback_fitted_result,
                path_key="model_bundle_path",
                consumer_step=WORKFLOW_STEPS[7],
                producer_product="fitted fallback Validation model bundle",
            )
            fallback_model_name = fallback_fitted_result["model_name"]
            fallback_model_type = fallback_fitted_result["model_type"]
        else:
            logger.info(
                "SKIP — Step 7/%d: fallback fitting is optional for primary model type %s.",
                TOTAL_WORKFLOW_STEPS,
                primary_model_type,
            )


        # ==================================================================
        # Step 8 — Compare and select Value Validation Model
        # ==================================================================

        value_selection_result = None
        value_selected_model_path = None

        if enabled_products["value_selected_model"]:
            value_selection_result = run_step(
                logger=logger,
                step_number=8,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[7],
                runner=lambda: compare_and_select_value_model(
                    cfg=cfg,
                    training_sample_path=training_sample_path,
                    force=force,
                    logger=logger,
                ),
            )
            value_selected_model_path = require_artifact(
                result=value_selection_result,
                path_key="value_selected_model_path",
                consumer_step=WORKFLOW_STEPS[8],
                producer_product="selected value Validation model",
            )

            value_selection_predictions_path = require_artifact(
                result=value_selection_result,
                path_key="value_model_predictions_path",
                consumer_step=WORKFLOW_STEPS[11],
                producer_product="value-model selection predictions",
            )        
        # ==================================================================
        # Step 9 — Fit Value Validation Model
        # ==================================================================
        value_fitted_model_result = None
        if enabled_products["value_fitted_model"]:
            value_fitted_model_result = run_step(
                logger=logger,
                step_number=9,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[8],
                runner=lambda: fit_selected_validation_model(
                    cfg=replace(cfg, model=cfg.value_model),
                    training_sample_path=training_sample_path,
                    selected_model_path=value_selected_model_path,
                    output_path=cfg.product_path("value_fitted_model"),
                    run_id=run_id,
                    force=force,
                    logger=logger,
                ),
            )

        value_fitted_model_path = require_artifact(
            result=value_fitted_model_result,
            path_key="fitted_model_path",
            consumer_step=WORKFLOW_STEPS[9],
            producer_product="fitted value Validation model",
        )

        value_model_bundle_path = require_artifact(
            result=value_fitted_model_result,
            path_key="model_bundle_path",
            consumer_step=WORKFLOW_STEPS[9],
            producer_product="fitted value Validation model bundle",
        )

        # ==================================================================
        # Step 10 — Run Validation inference
        # ==================================================================

        inference_result = run_step(
            logger=logger,
            step_number=10,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[9],
            runner=lambda: run_validation_inference(
                cfg=cfg,
                input_path=training_sample_path,
                fitted_model_path=fitted_model_path,
                model_bundle_path=model_bundle_path,
                fallback_fitted_model_path=fallback_fitted_model_path,
                fallback_model_bundle_path=fallback_model_bundle_path,
                value_fitted_model_path=value_fitted_model_path,
                value_model_bundle_path=value_model_bundle_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        inference_path = require_artifact(
            result=inference_result,
            path_key="inference_path",
            consumer_step=WORKFLOW_STEPS[10],
            producer_product="Validation inference",
        )

        # ==================================================================
        # Step 11 — Audit Validation inference
        # ==================================================================

        inference_audit_result = run_step(
            logger=logger,
            step_number=11,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[10],
            runner=lambda: audit_validation_inference(
                cfg=cfg,
                inference_path=inference_path,
                primary_model_name=fitted_model_result["model_name"],
                primary_model_type=primary_model_type,
                fallback_model_name=fallback_model_name,
                fallback_model_type=fallback_model_type,
                value_model_name=value_fitted_model_result["model_name"],
                value_model_type=value_fitted_model_result["model_type"],
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        inference_audit_path = require_artifact(
            result=inference_audit_result,
            path_key="inference_audit_path",
            consumer_step=WORKFLOW_STEPS[11],
            producer_product="Validation inference audit",
        )


        # ==================================================================
        # Step 12 — Build Quantity Plausibility
        # ==================================================================
        plausibility_result = run_step(
            logger=logger,
            step_number=12,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[11],
            runner=lambda: build_plausibility(
                cfg=cfg,
                training_sample_path=training_sample_path,
                inference_path=inference_path,
                primary_model_name=fitted_model_result["model_name"],
                value_model_name=value_fitted_model_result["model_name"],
                value_selection_predictions_path=value_selection_predictions_path,
                force=force,
                logger=logger,
            ),
        )

        plausibility_path = require_artifact(
            result=plausibility_result,
            path_key="plausibility_path",
            consumer_step=WORKFLOW_STEPS[12],
            producer_product="Validation plausibility",
        )

        # ==================================================================
        # Step 13 — Evaluate release readiness
        # ==================================================================
        release_readiness_result = run_step(
            logger=logger,
            step_number=13,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[12],
            runner=lambda: evaluate_release_readiness(
                cfg=cfg,
                prepared_input_path=prepared_path,
                inference_path=inference_path,
                structural_audit_path=structural_audit_path,
                inference_audit_path=inference_audit_path,
                selected_model_path=selected_model_path,
                value_selected_model_path=value_selected_model_path,
                plausibility_path=plausibility_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        release_readiness_path = require_artifact(
            result=release_readiness_result,
            path_key="release_readiness_path",
            consumer_step=WORKFLOW_STEPS[12],
            producer_product="Validation release readiness",
        )

        ready_for_release = release_readiness_result["ready_for_release"]

        # ==================================================================
        # Step 14 — Build diagnostics
        # ==================================================================

        diagnostic_result = run_step(
            logger=logger,
            step_number=14,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[13],
            runner=lambda: build_diagnostics(
                cfg=cfg,
                input_path=training_sample_path,
                structural_audit_path=structural_audit_path,
                model_predictions_path=model_predictions_path,
                selected_model_path=selected_model_path,
                inference_path=inference_path,
                inference_audit_path=inference_audit_path,
                release_readiness_path=release_readiness_path,
                value_model_predictions_path=value_selection_predictions_path,
                plausibility_path=plausibility_path,
                value_selected_model_path=value_selected_model_path,
                value_fitted_model_path=value_fitted_model_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        diagnostics_path = require_artifact(
            result=diagnostic_result,
            path_key="diagnostics_path",
            consumer_step=WORKFLOW_STEPS[13],
            producer_product="quantity-model diagnostics",
        )

        # ==================================================================
        # Step 15 — Build Validation report
        # ==================================================================

        report_result = run_step(
            logger=logger,
            step_number=15,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[14],
            runner=lambda: build_validation_report(
                cfg=cfg,
                prepared_input_path=prepared_path,
                structural_audit_path=structural_audit_path,
                quantity_plausibility_result=plausibility_result,
                diagnostics_path=diagnostics_path,
                release_readiness_path=release_readiness_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )


        # ==============================================================
        # Final Validation readiness
        # ==============================================================

        structural_passed = (structural_result.get("audit_passed") is True)
        inference_passed = (inference_audit_result.get("audit_passed") is True)

        report_path = require_artifact(
            result=report_result,
            path_key="report_path",
            consumer_step="VACODA Release",
            producer_product="Validation report",
        )

        if ready_for_release:
            logger.info(
                "VALIDATION PASSED — READY FOR RELEASE")
        else:
            logger.warning(
                "VALIDATION COMPLETE — NOT READY FOR RELEASE",
            )

        workflow_elapsed = perf_counter() - workflow_start

        separator(logger)

        logger.info(
            "Validation workflow finished.",
        )
        logger.info("Run ID: %s", run_id)
 
        logger.info(
            "Decision report: %s",
            report_path,
        )

        latest_report_path = report_result.get(
            "latest_report_path"
        )

        if latest_report_path is not None:
            logger.info(
                "Latest decision report: %s",
                latest_report_path,
            )

        logger.info(
            "Decision: %s",
            ready_for_release,
        )
        logger.info(
            "Ready for release: %s",
            ready_for_release,
        )
        logger.info(
            "Workflow duration: %.2fs",
            workflow_elapsed,
        )

        separator(logger)


    return {
        "status": "completed",
        "run_id": run_id,
        "ready_for_release": ready_for_release,

        "output_path": inference_path,
        "inference_path": inference_path,
        "prepared_input_path": prepared_path,
        "structural_audit_path": structural_audit_path,
        "inference_audit_path": inference_audit_path,
        "diagnostics_path": diagnostics_path,
        "report_path": report_path,
        "selected_fallback_model_path": selected_fallback_model_path,
        "fallback_fitted_model_path": fallback_fitted_model_path,
        "fallback_model_bundle_path": fallback_model_bundle_path,

        "structural_audit_passed": structural_passed,
        "inference_audit_passed": inference_passed,
        "release_readiness_path": release_readiness_path,
        "release_failure_reasons": release_readiness_result["failure_reasons"],
        "workflow_duration_seconds": workflow_elapsed,
    }
