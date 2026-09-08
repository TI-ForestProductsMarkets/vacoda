"""
vacoda.steps.validation.workflow

Orchestrate final VACODA validation, quality control, and release.

This module intentionally contains orchestration only. The validation,
statistical, imputation, and release logic belongs in the individual
Validation step modules.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from vacoda.config.steps.validation import ValidationConfig

from vacoda.steps.validation.prepare_input import prepare_validation_input
from vacoda.steps.validation.structural import  audit_structural_integrity
from vacoda.steps.validation.model import  develop_quantity_model
# from vacoda.steps.validation.consistency import  audit_cross_variable_consistency
# from vacoda.steps.validation.finalize import build_validated_trade
# from vacoda.steps.validation.release import build_release_dataset
from vacoda.steps.validation.reporting import build_validation_report

from vacoda.utils.artifacts import (
    require_artifact,
    require_enabled_products,
)
from vacoda.utils.display import get_symbols
from vacoda.utils.runtime import (
    require_result_path,
    run_step,
    separator,
    timed_workflow,
)


REQUIRED_PRODUCTS = (
    "prepared_input",
    "structural_audit",
    "model_diagnostics",
    "consistency_audit",
    "validated_trade",
    "release",
)


WORKFLOW_STEPS = [
    "Prepare Validation input",
    "Audit structural integrity",
    "Build Model diagnostics",
    # "Build validated trade",
    # "Audit cross-variable consistency",
    # "Build VACODA release dataset",
    "Write Validation decision report",
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
        # Step 3 — Quantity-model diagnostics
        # ==================================================================

        model_diagnostic_result = run_step(
            logger=logger,
            step_number=3,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[2],
            runner=lambda: develop_quantity_model(
                cfg=cfg,
                input_path=prepared_path,
                structural_audit_path=structural_audit_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        model_diagnostics_path = require_artifact(
            result=model_diagnostic_result,
            path_key="model_diagnostics_path",
            consumer_step=WORKFLOW_STEPS[3],
            producer_product="quantity-model diagnostics",
        )
    #     # ==================================================================
    #     # Step 6 — Cross-variable consistency
    #     # ==================================================================

    #     consistency_result = run_step(
    #         logger=logger,
    #         step_number=6,
    #         total_steps=TOTAL_WORKFLOW_STEPS,
    #         title=WORKFLOW_STEPS[5],
    #         runner=lambda: audit_cross_variable_consistency(
    #             cfg=cfg,
    #             input_path=prepared_path,
    #             run_id=run_id,
    #             force=force,
    #             logger=logger,
    #         ),
    #     )

    #     # ==================================================================
    #     # Step 7 — Build validated trade and provenance
    #     # ==================================================================

    #     validated_result = run_step(
    #         logger=logger,
    #         step_number=7,
    #         total_steps=TOTAL_WORKFLOW_STEPS,
    #         title=WORKFLOW_STEPS[6],
    #         runner=lambda: build_validated_trade(
    #             cfg=cfg,
    #             input_path=prepared_path,
    #             structural_audit_path=structural_audit_path,
    #             missingness_path=missingness_path,
    #             unit_value_diagnostics_path=unit_value_path,
    #             outliers_path=outliers_path,
    #             consistency_audit_path=consistency_audit_path,
    #             run_id=run_id,
    #             force=force,
    #             logger=logger,
    #         ),
    #     )


    #     # ==================================================================
    #     # Step 8 — Build final VACODA release
    #     # ==================================================================

    #     release_result = run_step(
    #         logger=logger,
    #         step_number=8,
    #         total_steps=TOTAL_WORKFLOW_STEPS,
    #         title=WORKFLOW_STEPS[7],
    #         runner=lambda: build_release_dataset(
    #             cfg=cfg,
    #             input_path=validated_trade_path,
    #             run_id=run_id,
    #             force=force,
    #             logger=logger,
    #         ),
    #     )

    #     release_path = require_artifact(
    #         result=release_result,
    #         path_key="release_path",
    #         consumer_step=WORKFLOW_STEPS[8],
    #         producer_product="VACODA release dataset",
    #     )

        # ==================================================================
        # Step 4 — Build Validation report
        # ==================================================================

        report_result = run_step(
            logger=logger,
            step_number=4,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[3],
            runner=lambda: build_validation_report(
                cfg=cfg,
                prepared_input_path=prepared_path,
                structural_audit_path=structural_audit_path,
                model_diagnostics_path=model_diagnostics_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

    #     report_path = require_result_path(
    #         report_result,
    #         "Validation reporting result did not contain "
    #         "an expected report path.",
    #         "report_path",
    #         "output_path",
    #     )

    #     if not report_path.is_file():
    #         raise FileNotFoundError(
    #             f"Validation decision report not found: {report_path}"
    #         )

    #     # ==================================================================
    #     # Final release decision
    #     # ==================================================================

    #     release_output_valid = bool(
    #         release_path.is_file()
    #         and release_path.stat().st_size > 0
    #     )

    #     decision = report_result.get("decision")

    #     ready_for_release = bool(
    #         structural_passed
    #         and consistency_passed
    #         and release_output_valid
    #         and decision == "APPROVED"
    #     )

    #     if not ready_for_release:
    #         raise RuntimeError(
    #             "Validation finished without an approved, usable "
    #             "VACODA release dataset."
    #         )

        workflow_elapsed = perf_counter() - workflow_start

        separator(logger)

        logger.info(
            "%s Validation workflow finished.",
            get_symbols("success"),
        )
        logger.info("Run ID: %s", run_id)
    #     logger.info(
    #         "Validated trade: %s",
    #         validated_trade_path,
    #     )
    #     logger.info(
    #         "Release dataset: %s",
    #         release_path,
    #     )
        # logger.info(
        #     "Structural audit passed: %s",
        #     structural_passed,
        # )
    #     logger.info(
    #         "Consistency audit passed: %s",
    #         consistency_passed,
    #     )
    #     logger.info(
    #         "Provenance: %s",
    #         provenance_path,
    #     )
    #     logger.info(
    #         "Decision report: %s",
    #         report_path,
    #     )

    #     latest_report_path = report_result.get(
    #         "latest_report_path"
    #     )

    #     if latest_report_path is not None:
    #         logger.info(
    #             "Latest decision report: %s",
    #             latest_report_path,
    #         )

    #     logger.info(
    #         "Decision: %s",
    #         decision,
    #     )
    #     logger.info(
    #         "Ready for release: %s",
    #         ready_for_release,
    #     )
    #     logger.info(
    #         "Workflow duration: %.2fs",
    #         workflow_elapsed,
    #     )

        separator(logger)

    return {
        "status": "completed",
        "run_id": run_id,

    #     # Master-pipeline readiness contract.
    #     "ready_for_release": ready_for_release,

    #     # Canonical primary output and audit.
    #     "output_path": release_path,
    #     "audit_path": consistency_audit_path,

    #     # Explicit Validation products.
         "prepared_input_path": prepared_path,
         "structural_audit_path": structural_result["structural_audit_path"],
    #     "consistency_audit_path": consistency_audit_path,
    #     "validated_trade_path": validated_trade_path,
    #     "release_path": release_path,
    #     "report_path": report_path,
    #     "latest_report_path": report_result.get(
    #         "latest_report_path"
    #     ),

    #     # Final Validation outcome.
    #     "structural_audit_passed": structural_passed,
    #     "consistency_audit_passed": consistency_passed,
    #     "decision": decision,

    #     "workflow_duration_seconds": workflow_elapsed,
     }