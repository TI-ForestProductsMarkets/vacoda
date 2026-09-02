"""
vacoda.steps.trade_sync.workflow

Orchestrate bilateral trade reconciliation after CIF/FOB correction.

This module intentionally contains orchestration only. The scientific
logic belongs in the individual Trade Sync step modules.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
from vacoda.config.steps.trade_sync import TradeSyncConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.runtime import require_result_path, run_step, separator
from vacoda.steps.trade_sync.prepare_input import  prepare_trade_sync_input
from vacoda.steps.trade_sync.build_pairs import build_trade_sync_pairs
from vacoda.steps.trade_sync.reliability_evidence import build_trade_sync_reliability_evidence
from vacoda.steps.trade_sync.country_reliability import estimate_trade_sync_country_reliability
from vacoda.steps.trade_sync.reconcile import reconcile_trade
from vacoda.steps.trade_sync.diagnostics import build_trade_sync_diagnostics
from vacoda.steps.trade_sync.reporting import build_trade_sync_report
from vacoda.utils.runtime import timed_workflow
from vacoda.utils.artifacts import require_artifact, require_product_dependency, require_enabled_products

REQUIRED_PRODUCTS = (
    "prepared_input",
    "bilateral_pairs",
    "reliability_evidence",
    "country_reliability",
    "reconciled_trade",
    "reconciliation_audit",
    "diagnostics",
)


WORKFLOW_STEPS = [
    "Prepare Trade Sync input",
    "Build Trade Sync bilateral pairs",
    "Build Trade Sync reliability evidence",
    "Estimate Trade Sync country reliability",
    "Reconcile trade",
    "Audit trade reconciliation",
    "Build Trade Sync diagnostics",
    "Write Trade Sync decision report",
]

TOTAL_WORKFLOW_STEPS = len(WORKFLOW_STEPS)


    
def validate_reconciliation_audit(
    *,
    reconciled_result: dict,
    logger,
) -> dict:
    """Validate the audit artifact produced during reconciliation."""

    audit_path = require_result_path(
            reconciled_result,
            "Trade Sync reconciliation result did not contain "
            "the reconciliation-audit path.",
            "reconciliation_audit_path",
            "audit_path",
        )

    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Trade Sync reconciliation audit not found: {audit_path}"
        )

    if not bool(reconciled_result.get("audit_passed", False)):
        raise RuntimeError(
            "Trade Sync reconciliation audit did not pass."
        )

    logger.info("  Audit: %s", audit_path)
    logger.info("  Audit passed: True")

    return {
        "status": "completed",
        "output_path": audit_path,
        "reconciliation_audit_path": audit_path,
        "audit_passed": True,
        "ready_for_diagnostics": True,
    }

def run_trade_sync_workflow(
    *,
    cfg: TradeSyncConfig,
    run_id: str,
    logger,
    force: bool = False,
    input_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run the complete Trade Sync workflow.

    Parameters
    ----------
    cfg:
        Parsed Trade Sync configuration.
    run_id:
        Master-pipeline run identifier.
    logger:
        Configured workflow logger.
    force:
        Ignore reusable artifacts inside Trade Sync step modules.
    input_path:
        Optional explicit CIF/FOB-corrected input. When omitted,
        ``cfg.input_path()`` is used.

    Expected step modules
    ---------------------
    vacoda.trade_sync.prepare_input
        prepare_trade_sync_input

    vacoda.trade_sync.build_pairs
        build_trade_sync_pairs

    vacoda.trade_sync.reconcile
        reconcile_trade

    vacoda.trade_sync.diagnostics
        build_trade_sync_diagnostics

    vacoda.trade_sync.report
        build_trade_sync_report
    """

    resolved_input_path = (
        Path(input_path)
        if input_path is not None
        else cfg.input_path()
    )

    if not resolved_input_path.is_file():
        raise FileNotFoundError(
            "Trade Sync input parquet not found: "
            f"{resolved_input_path}"
        )

    enabled_products = cfg.enabled_products()

    require_enabled_products(
        workflow_name="Trade Sync",
        enabled_products=enabled_products,
        required_products=REQUIRED_PRODUCTS,
    )

    require_product_dependency(
        workflow_name="Trade Sync",
        enabled_products=enabled_products,
        product="decision_report",
        required_product="diagnostics",
    )

    cfg.output_dir().mkdir(
        parents=True,
        exist_ok=True,
    )

    workflow_start = perf_counter()


    with timed_workflow(logger, "Trade Sync"):

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
            "Reconciliation method: %s",
            cfg.method.reconciliation.method,
        )
        logger.info(
            "Retain unmatched flows: %s",
            cfg.method.reconciliation.retain_unmatched_flows,
        )
        logger.info(
            "Agreement tolerance enabled: %s",
            cfg.diagnostics.agreement_tolerance.enabled,
        )
        logger.info(
            "Enabled products: %s",
            " ".join(
                f"{name}={enabled}"
                for name, enabled in enabled_products.items()
            ),
        )  

        # ==================================================================
        # Step 1 — Prepare canonical Trade Sync input
        # ==================================================================

        prepared_result = run_step(
                logger=logger,
                step_number=1,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[0],
                runner=lambda: prepare_trade_sync_input(
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
                producer_product="prepared Trade Sync input",
            )

        # ==================================================================
        # Step 2 — Build bilateral import/export pairs
        # ==================================================================

        pairs_result = run_step(
                logger=logger,
                step_number=2,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[1],
                runner=lambda: build_trade_sync_pairs(
                    cfg=cfg,
                    input_path=prepared_path,
                    run_id=run_id,
                    force=force,
                    logger=logger,
                ),
            )

        pairs_path = require_artifact(
            result=pairs_result,
            path_key="bilateral_pairs_path",
            consumer_step=WORKFLOW_STEPS[2],
            producer_product="bilateral-pairs Parquet",
        )
        

        # ==================================================================
        # Step 3 — Build reliability evidence
        # ==================================================================

        reliability_evidence_result  = run_step(
                logger=logger,
                step_number=3,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[2],
                runner=lambda: build_trade_sync_reliability_evidence(
                    cfg=cfg,
                    input_path=prepared_path,
                    run_id=run_id,
                    force=force,
                    logger=logger,
                ),
            )

        reliability_evidence_path = require_artifact(
            result=reliability_evidence_result,
            path_key="reliability_evidence_path",
            consumer_step=WORKFLOW_STEPS[3],
            producer_product="reliability-evidence Parquet",
        )

        # ==================================================================
        # Step 4 — Estimate country reliability
        # ==================================================================
        country_reliability_result  = run_step(
                logger=logger,
                step_number=4,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[3],
                runner=lambda: estimate_trade_sync_country_reliability(
                    cfg=cfg,
                    input_path=reliability_evidence_path,
                    run_id=run_id,
                    force=force,
                    logger=logger,
                ),
            )

        country_reliability_path = require_artifact(
            result=country_reliability_result,
            path_key="country_reliability_path",
            consumer_step=WORKFLOW_STEPS[4],
            producer_product="country-reliability Parquet",
        )


        # ==================================================================
        # Step 5 — Reconcile bilateral trade values
        # ==================================================================

        reconciled_result = run_step(
                logger=logger,
                step_number=5,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[4],
                runner=lambda: reconcile_trade(
                    cfg=cfg,
                    prepared_input_path=prepared_path,
                    reliability_evidence_path=reliability_evidence_path,
                    country_reliability_path=country_reliability_path,
                    run_id=run_id,
                    force=force,
                    logger=logger,
                ),
            )

        reconciled_path = require_artifact(
            result=reconciled_result,
            path_key="reconciled_trade_path",
            consumer_step=WORKFLOW_STEPS[5],
            producer_product="reconciled-trade Parquet",
        )

        # ==================================================================
        # Step 6 — Validate reconciliation audit
        # ==================================================================

        audit_result = run_step(
                logger=logger,
                step_number=6,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[5],
                runner=lambda: validate_reconciliation_audit(
                    reconciled_result=reconciled_result,
                    logger=logger,
                ),
            )

        audit_path = require_artifact(
            result=audit_result,
            path_key="reconciliation_audit_path",
            consumer_step=WORKFLOW_STEPS[6],
            producer_product="reconciliation-audit Parquet",
        )

        audit_passed = bool(
                audit_result.get(
                    "audit_passed",
                    False,
                )
            )
        if not audit_passed:
            raise RuntimeError(
                "Trade Sync reconciliation_audit product must be enabled."
            )


        # ==================================================================
        # Step 7 — Build diagnostics
        # ==================================================================

        diagnostics_result = run_step(
                logger=logger,
                step_number=7,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[6],
                runner=lambda: build_trade_sync_diagnostics(
                    cfg=cfg,
                    reconciled_trade_path=reconciled_path,
                    country_reliability_path=country_reliability_path,
                    reconciliation_audit_path=audit_path,
                    run_id=run_id,
                    force=force,
                    logger=logger,
                ),
            )

        diagnostics_path = require_artifact(
            result=diagnostics_result,
            path_key="diagnostics_path",
            consumer_step=WORKFLOW_STEPS[7],
            producer_product="diagnostics Parquet",
        )


        # ==================================================================
        # Step 8 — Write Trade Sync decision report
        # ==================================================================

        report_result = run_step(
                logger=logger,
                step_number=8,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[7],
                runner=lambda: build_trade_sync_report(
                    cfg=cfg,
                    reconciled_trade_path=reconciled_path,
                    country_reliability_path=country_reliability_path,
                    reconciliation_audit_path=audit_path,
                    diagnostics_path=diagnostics_path,
                    run_id=run_id,
                    force=force,
                    logger=logger,
                ),
            )

        report_path = require_result_path(
                report_result,
                "Trade Sync reporting result did not contain "
                "an expected report path.",
                "report_path",
                "output_path",
            )

        if not report_path.is_file():
            raise FileNotFoundError(
                f"Trade Sync decision report not found: {report_path}"
            )
        
        reconciled_output_valid = bool(
            reconciled_path.is_file()
            and reconciled_path.stat().st_size > 0
        )

        ready_for_value_sync = bool(
            audit_passed
            and reconciled_output_valid
        )

        if not ready_for_value_sync:
            raise RuntimeError(
                "Trade Sync finished without an audited, usable "
                "reconciled-trade output."
            )

        n_failed_audit_rows = int(
            reconciled_result.get(
                "n_failed_audit_rows",
                0,
            )
        )

        ########################################################################################
        
        workflow_elapsed = perf_counter() - workflow_start

        separator(logger)
        logger.info(
                "%s Trade Sync workflow finished.",
                get_symbols("success"),
            )
        logger.info(
                "Run ID: %s",
                run_id,
            )
        logger.info(
                "Reconciled trade: %s",
                reconciled_path,
            )
        logger.info(
                "Reconciliation audit: %s",
                audit_path,
            )
        logger.info(
                "Reconciliation audit passed: %s",
                audit_passed,
            )
        logger.info(
                "Failed audit rows: %s",
                n_failed_audit_rows,
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
                "Ready for Value Sync: %s",
                ready_for_value_sync,
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
            "ready_for_value_sync": ready_for_value_sync,

            # Canonical primary output and audit keys used by the master
            # pipeline and checkpoint system.
            "output_path": reconciled_path,
            "audit_path": audit_path,

            # Explicit Trade Sync product paths.
            "prepared_input_path": prepared_path,
            "bilateral_pairs_path": pairs_path,
            "reliability_evidence_path": reliability_evidence_path,
            "country_reliability_path": country_reliability_path,
            "reconciled_trade_path": reconciled_path,
            "reconciliation_audit_path": audit_path,
            "diagnostics_path": diagnostics_path,
            "report_path": report_path,
            "latest_report_path": report_result.get(
                "latest_report_path"
            ),

            # Final validation outcome.
            "audit_passed": audit_passed,
            "n_failed_audit_rows": n_failed_audit_rows,

            # Useful final decision metadata.
            "decision": report_result.get("decision"),
            "workflow_duration_seconds": workflow_elapsed,
        }