"""
vacoda.quantity.workflow

Orchestrate quantity synchronization after HS synchronization.

This module intentionally contains orchestration only. The scientific
logic belongs in the individual Quantity step modules.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from vacoda.config.quantity import QuantityConfig
from vacoda.quantity.diagnostics import build_quantity_diagnostics
from vacoda.quantity.prepare_input import prepare_quantity_input
from vacoda.quantity.reconciliation_evidence import (
    build_quantity_reconciliation_evidence,
)
from vacoda.quantity.reporting import build_quantity_report
from vacoda.quantity.synchronize import synchronize_quantity
from vacoda.utils.artifacts import (
    require_artifact,
    require_enabled_products,
    require_product_dependency,
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
    "reconciliation_evidence",
    "synchronized_quantity",
    "diagnostics",
)


WORKFLOW_STEPS = [
    "Prepare Quantity input",
    "Build Quantity reconciliation evidence",
    "Synchronize quantity",
    "Build Quantity diagnostics",
    "Write Quantity decision report",
]


TOTAL_WORKFLOW_STEPS = len(WORKFLOW_STEPS)


def run_quantity_workflow(
    *,
    cfg: QuantityConfig,
    run_id: str,
    logger,
    force: bool = False,
    input_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run the complete Quantity workflow.

    Parameters
    ----------
    cfg:
        Parsed Quantity configuration.
    run_id:
        Master-pipeline run identifier.
    logger:
        Configured workflow logger.
    force:
        Ignore reusable artifacts inside Quantity step modules.
    input_path:
        Optional explicit HS-synchronized input. When omitted,
        ``cfg.input_path()`` is used.

    Expected step modules
    ---------------------
    vacoda.quantity.prepare_input
        prepare_quantity_input

    vacoda.quantity.reconciliation_evidence
        build_quantity_reconciliation_evidence

    vacoda.quantity.synchronize
        synchronize_quantity

    vacoda.quantity.diagnostics
        build_quantity_diagnostics

    vacoda.quantity.reporting
        build_quantity_report
    """

    resolved_input_path = (
        Path(input_path)
        if input_path is not None
        else cfg.input_path()
    )

    if not resolved_input_path.is_file():
        raise FileNotFoundError(
            "Quantity input parquet not found: "
            f"{resolved_input_path}"
        )

    enabled_products = cfg.enabled_products()

    require_enabled_products(
        workflow_name="Quantity",
        enabled_products=enabled_products,
        required_products=REQUIRED_PRODUCTS,
    )

    require_product_dependency(
        workflow_name="Quantity",
        enabled_products=enabled_products,
        product="decision_report",
        required_product="diagnostics",
    )

    cfg.output_dir().mkdir(
        parents=True,
        exist_ok=True,
    )

    workflow_start = perf_counter()

    with timed_workflow(logger, "Quantity"):

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
            "Quantity unit for kg reconciliation: %s",
            cfg.method.kg_reconciliation.quantity_unit,
        )
        logger.info(
            "Require positive values: %s",
            cfg.method.kg_reconciliation.require_positive_values,
        )
        logger.info(
            "Source-selection method: %s",
            cfg.method.kg_reconciliation.source_selection.method,
        )
        logger.info(
            "Agreement tolerance: %.6f",
            cfg.method.kg_reconciliation.agreement.relative_tolerance,
        )
        logger.info(
            "Use gross weight as quantity proxy: %s",
            cfg.method.gross_weight.use_as_quantity_proxy,
        )
        logger.info(
            "Enabled products: %s",
            " ".join(
                f"{name}={enabled}"
                for name, enabled in enabled_products.items()
            ),
        )

        # ==================================================================
        # Step 1 — Prepare canonical Quantity input
        # ==================================================================

        prepared_result = run_step(
            logger=logger,
            step_number=1,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[0],
            runner=lambda: prepare_quantity_input(
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
            producer_product="prepared Quantity input",
        )

        # ==================================================================
        # Step 2 — Build reconciliation evidence
        # ==================================================================

        evidence_result = run_step(
            logger=logger,
            step_number=2,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[1],
            runner=lambda: build_quantity_reconciliation_evidence(
                cfg=cfg,
                input_path=prepared_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        evidence_path = require_artifact(
            result=evidence_result,
            path_key="reconciliation_evidence_path",
            consumer_step=WORKFLOW_STEPS[2],
            producer_product="quantity reconciliation-evidence Parquet",
        )

        # ==================================================================
        # Step 3 — Synchronize quantity
        # ==================================================================

        synchronized_result = run_step(
            logger=logger,
            step_number=3,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[2],
            runner=lambda: synchronize_quantity(
                cfg=cfg,
                prepared_input_path=prepared_path,
                reconciliation_evidence_path=evidence_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        synchronized_path = require_artifact(
            result=synchronized_result,
            path_key="synchronized_quantity_path",
            consumer_step=WORKFLOW_STEPS[3],
            producer_product="synchronized Quantity Parquet",
        )

        # ==================================================================
        # Step 4 — Build diagnostics
        # ==================================================================

        diagnostics_result = run_step(
            logger=logger,
            step_number=4,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[3],
            runner=lambda: build_quantity_diagnostics(
                cfg=cfg,
                prepared_input_path=prepared_path,
                reconciliation_evidence_path=evidence_path,
                synchronized_quantity_path=synchronized_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        diagnostics_path = require_artifact(
            result=diagnostics_result,
            path_key="diagnostics_path",
            consumer_step=WORKFLOW_STEPS[4],
            producer_product="Quantity diagnostics Parquet",
        )

        # ==================================================================
        # Step 5 — Write Quantity decision report
        # ==================================================================

        report_result = run_step(
            logger=logger,
            step_number=5,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[4],
            runner=lambda: build_quantity_report(
                cfg=cfg,
                prepared_input_path=prepared_path,
                reconciliation_evidence_path=evidence_path,
                synchronized_quantity_path=synchronized_path,
                diagnostics_path=diagnostics_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        report_path = require_result_path(
            report_result,
            "Quantity reporting result did not contain "
            "an expected report path.",
            "report_path",
            "output_path",
        )

        if not report_path.is_file():
            raise FileNotFoundError(
                f"Quantity decision report not found: {report_path}"
            )

        synchronized_output_valid = bool(
            synchronized_path.is_file()
            and synchronized_path.stat().st_size > 0
        )

        ready_for_cif_fob = synchronized_output_valid

        if not ready_for_cif_fob:
            raise RuntimeError(
                "Quantity workflow finished without a usable "
                "synchronized-quantity output."
            )

        workflow_elapsed = perf_counter() - workflow_start

        separator(logger)
        logger.info(
            "%s Quantity workflow finished.",
            get_symbols("success"),
        )
        logger.info("Run ID: %s", run_id)
        logger.info(
            "Prepared input: %s",
            prepared_path,
        )
        logger.info(
            "Reconciliation evidence: %s",
            evidence_path,
        )
        logger.info(
            "Synchronized quantity: %s",
            synchronized_path,
        )

        if diagnostics_path is not None:
            logger.info(
                "Diagnostics: %s",
                diagnostics_path,
            )

        if report_path is not None:
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
            "Ready for CIF/FOB: %s",
            ready_for_cif_fob,
        )
        logger.info(
            "Workflow duration: %.2fs",
            workflow_elapsed,
        )
        separator(logger)

    return {
        "status": "completed",
        "run_id": run_id,

        # Master-pipeline readiness contract.
        "ready_for_cif_fob": ready_for_cif_fob,

        # Canonical primary output.
        "output_path": synchronized_path,

        # Explicit Quantity product paths.
        "prepared_input_path": prepared_path,
        "reconciliation_evidence_path": evidence_path,
        "synchronized_quantity_path": synchronized_path,
        "diagnostics_path": diagnostics_path,
        "report_path": report_path,
        "latest_report_path": report_result.get(
            "latest_report_path"
        ),

        # Useful final decision metadata.
        "decision": report_result.get("decision"),
        "workflow_duration_seconds": workflow_elapsed,
    }