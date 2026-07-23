"""
vacoda.trade_sync.workflow

Orchestrate bilateral trade reconciliation after CIF/FOB correction.

This module intentionally contains orchestration only. The scientific
logic belongs in the individual Trade Sync step modules.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
from vacoda.config.trade_sync import TradeSyncConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.logger import require_result_path, run_step, separator
from vacoda.trade_sync.prepare_input import  prepare_trade_sync_input
from vacoda.trade_sync.build_pairs import build_trade_sync_pairs
from vacoda.trade_sync.reliability_evidence import build_trade_sync_reliability_evidence
from vacoda.trade_sync.country_reliability import estimate_trade_sync_country_reliability
from vacoda.trade_sync.reconcile import reconcile_trade
from vacoda.trade_sync.diagnostics import build_trade_sync_diagnostics
from vacoda.trade_sync.reporting import build_trade_sync_report

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
        write_trade_sync_decision_report
    """
    info = get_symbols("info")
    ok = get_symbols("check")
    warning = get_symbols("warning")

    resolved_input_path = (
        Path(input_path)
        if input_path is not None
        else cfg.input_path()
    )

    if not resolved_input_path.exists():
        raise FileNotFoundError(
            "Trade Sync input parquet not found: "
            f"{resolved_input_path}"
        )

    cfg.output_dir().mkdir(
        parents=True,
        exist_ok=True,
    )

    enabled_products = cfg.enabled_products()
    total_steps = TOTAL_WORKFLOW_STEPS

    separator(logger)
    logger.info("%s START — Trade Sync workflow", info)
    separator(logger)
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

    workflow_start = perf_counter()

    # ==================================================================
    # Step 1 — Prepare canonical Trade Sync input
    # ==================================================================

    if enabled_products["prepared_input"]:

        prepared_result = run_step(
            logger=logger,
            step_number=1,
            total_steps=total_steps,
            title="Prepare Trade Sync input",
            runner=lambda: prepare_trade_sync_input(
                cfg=cfg,
                input_path=resolved_input_path,
                force=force,
                logger=logger,
            ),
        )

        prepared_path = require_result_path(
            prepared_result,
            "Trade Sync step result did not contain any expected path key",
            "prepared_input_path",
            "output_path",
        )
    else:
        logger.warning(
            "%s Trade Sync prepared-input product is disabled; "
            "using the workflow input directly.",
            warning,
        )
        prepared_result = {
            "status": "disabled",
            "prepared_input_path": resolved_input_path,
        }
        prepared_path = resolved_input_path

    # ==================================================================
    # Step 2 — Build bilateral import/export pairs
    # ==================================================================

    if enabled_products["bilateral_pairs"]:

        pairs_result = run_step(
            logger=logger,
            step_number=2,
            total_steps=total_steps,
            title="Build Trade Sync bilateral pairs",
            runner=lambda: build_trade_sync_pairs(
                cfg=cfg,
                input_path=prepared_path,
                force=force,
                logger=logger,
            ),
        )

        pairs_path = require_result_path(
            pairs_result,
            "Trade Sync step result did not contain any expected path key",
            "bilateral_pairs_path",
            "output_path",
        )
    else:
        raise RuntimeError(
            "Trade Sync bilateral_pairs must be enabled because "
            "reconciliation depends on paired import/export evidence."
        )
    

    # ==================================================================
    # Step 3 — Build reliability evidence
    # ==================================================================

    if enabled_products["reliability_evidence"]:

        reliability_evidence_result  = run_step(
            logger=logger,
            step_number=3,
            total_steps=total_steps,
            title="Build Trade Sync reliability evidence",
            runner=lambda: build_trade_sync_reliability_evidence(
                cfg=cfg,
                input_path=prepared_path,
                force=force,
                logger=logger,
                ),
            )

        reliability_evidence_path = require_result_path(
                    reliability_evidence_result,
                    "Trade Sync reliability-evidence result did not contain an expected path key",
                    "reliability_evidence_path",
                    "output_path",
                )
    else:
        raise RuntimeError(
            "Trade Sync reliability_evidence must be enabled."
        )
    
    # ==================================================================
    # Step 4 — Estimate country reliability
    # ==================================================================
    if enabled_products["country_reliability"]:

        country_reliability_result  = run_step(
            logger=logger,
            step_number=4,
            total_steps=total_steps,
            title="Estimate Trade Sync country reliability",
            runner=lambda: estimate_trade_sync_country_reliability(
                cfg=cfg,
                input_path=reliability_evidence_path,
                force=force,
                logger=logger,
            ),
        )

        country_reliability_path = require_result_path(
                    country_reliability_result,
                    "Trade Sync country-reliability result did not contain an expected path key",
                    "country_reliability_path",
                    "output_path",
        )
    else:
        raise RuntimeError(
            "Trade Sync country_reliability must be enabled."
        )

    # ==================================================================
    # Step 5 — Reconcile bilateral trade values
    # ==================================================================

    if enabled_products["reconciled_trade"]:

        reconciled_result = run_step(
            logger=logger,
            step_number=5,
            total_steps=total_steps,
            title="Reconcile trade",
            runner=lambda: reconcile_trade(
                cfg=cfg,
                prepared_input_path=prepared_path,
                reliability_evidence_path=reliability_evidence_path,
                country_reliability_path=country_reliability_path,
                force=force,
                logger=logger,
            ),
        )

        reconciled_path = require_result_path(
            reconciled_result,
            "Trade Sync reconciliation result did not contain "
            "an expected output path.",
            "reconciled_trade_path",
            "output_path",
        )

    else:
        raise RuntimeError(
            "Trade Sync reconciled_trade product must be enabled."
        )


    # ==================================================================
    # Step 6 — Validate reconciliation audit
    # ==================================================================

    if enabled_products["reconciliation_audit"]:
        audit_result = run_step(
            logger=logger,
            step_number=6,
            total_steps=total_steps,
            title="Audit trade reconciliation",
            runner=lambda: validate_reconciliation_audit(
                reconciled_result=reconciled_result,
                logger=logger,
            ),
        )

        audit_path = require_result_path(
            audit_result,
            "Trade Sync audit-validation result did not contain the reconciliation-audit path.",
            "reconciliation_audit_path",
            "output_path",
        )

        audit_passed = bool(
            audit_result.get(
                "audit_passed",
                False,
            )
        )
    else:
        raise RuntimeError(
            "Trade Sync reconciliation_audit product must be enabled."
        )


    # ==================================================================
    # Step 7 — Build diagnostics
    # ==================================================================

    if enabled_products["diagnostics"]:

        diagnostics_result = run_step(
            logger=logger,
            step_number=7,
            total_steps=total_steps,
            title="Build Trade Sync diagnostics",
            runner=lambda: build_trade_sync_diagnostics(
                cfg=cfg,
                reconciled_trade_path=reconciled_path,
                country_reliability_path=country_reliability_path,
                reconciliation_audit_path=audit_path,
                force=force,
                logger=logger,
            ),
        )

        diagnostics_path = require_result_path(
            diagnostics_result,
            "Trade Sync diagnostics result did not contain "
            "an expected path key.",
            "diagnostics_path",
            "output_path",
        )

    else:
        logger.info(
            "SKIP — Trade Sync diagnostics are disabled."
        )

        diagnostics_result = {
            "status": "disabled",
            "output_path": None,
            "diagnostics_path": None,
            "ready_for_decision_report": False,
        }

        diagnostics_path = None

    # ==================================================================
    # Step 8 — Write Trade Sync decision report
    # ==================================================================

    if cfg.artifacts.enable and diagnostics_path is None:
        raise RuntimeError(
            "Trade Sync decision reporting requires diagnostics, "
            "but the diagnostics product is disabled."
        )

    if cfg.artifacts.enable:

        report_result = run_step(
            logger=logger,
            step_number=8,
            total_steps=total_steps,
            title="Write Trade Sync decision report",
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

    else:
        logger.info(
            "SKIP — Trade Sync decision report is disabled."
        )

        report_result = {
            "status": "disabled",
            "report_path": None,
            "ready_for_processed_export": False,
        }

        report_path = None

    workflow_elapsed = perf_counter() - workflow_start

    reconciled_output_valid = bool(
        reconciled_path.is_file()
        and reconciled_path.stat().st_size > 0
    )

    ready_for_weight_sync = bool(
        audit_passed
        and reconciled_output_valid
    )

    if not ready_for_weight_sync:
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

    separator(logger)
    logger.info(
        "%s Trade Sync workflow finished.",
        ok,
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
        "Ready for Weight Sync: %s",
        ready_for_weight_sync,
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
        "ready_for_weight_sync": ready_for_weight_sync,

        # Compatibility alias until pipeline.yaml and the master pipeline
        # use ready_for_weight_sync directly.
        "ready_for_processed_export": ready_for_weight_sync,

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