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
# from vacoda.trade_sync.build_pairs import build_trade_sync_pairs
# from vacoda.trade_sync.reconcile import reconcile_trade
# from vacoda.trade_sync.diagnostics import build_trade_sync_diagnostics
# from vacoda.trade_sync.report import   write_trade_sync_decision_report

WORKFLOW_STEPS = [
    "Prepare Trade Sync input",
    "Build Trade Sync bilateral pairs",
    "Reconcile trade",
    "Build Trade Sync diagnostics",
    "Write Trade Sync decision report",
]

TOTAL_WORKFLOW_STEPS = len(WORKFLOW_STEPS)

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
        cfg.method.reconciliation.agreement_tolerance.enabled,
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
            "prepared_input_path",
            "output_path",
            error_message="Trade Sync step result did not contain any expected path key"
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
            "bilateral_pairs_path",
            "output_path",
        )
    else:
        raise RuntimeError(
            "Trade Sync bilateral_pairs must be enabled because "
            "reconciliation depends on paired import/export evidence."
        )

    # ==================================================================
    # Step 3 — Reconcile bilateral trade values
    # ==================================================================

    if enabled_products["reconciled_trade"]:

        reconciled_result = run_step(
            logger=logger,
            step_number=3,
            total_steps=total_steps,
            title="Reconcile bilateral trade",
            runner=lambda: reconcile_trade(
                cfg=cfg,
                input_path=pairs_path,
                force=force,
                logger=logger,
            ),
        )

        reconciled_path = require_result_path(
            reconciled_result,
            "reconciled_trade_path",
            "output_path",
        )
    else:
        raise RuntimeError(
            "Trade Sync reconciled_trade product must be enabled."
        )

    audit_path = Path(
        reconciled_result.get(
            "audit_path",
            cfg.product_path("reconciliation_audit"),
        )
    )

    audit_passed = bool(
        reconciled_result.get("audit_passed", False)
    )

    if not audit_passed:
        raise RuntimeError(
            "Trade Sync reconciliation audit did not pass."
        )

    # ==================================================================
    # Step 4 — Build diagnostics
    # ==================================================================

    if enabled_products["diagnostics"]:

        diagnostics_result = run_step(
            logger=logger,
            step_number=4,
            total_steps=total_steps,
            title="Build Trade Sync diagnostics",
            runner=lambda: build_trade_sync_diagnostics(
                cfg=cfg,
                reconciled_trade_path=reconciled_path,
                bilateral_pairs_path=pairs_path,
                force=force,
                logger=logger,
            ),
        )
    else:
        logger.info(
            "SKIP — Trade Sync diagnostics are disabled."
        )
        diagnostics_result = {
            "status": "disabled",
            "diagnostics_path": None,
        }

    # ==================================================================
    # Step 5 — Decision report
    # ==================================================================

    if cfg.decision_report.enable:

        report_result = run_step(
            logger=logger,
            step_number=5,
            total_steps=total_steps,
            title="Write Trade Sync decision report",
            runner=lambda: write_trade_sync_decision_report(
                cfg=cfg,
                run_id=run_id,
                prepared_result=prepared_result,
                pairs_result=pairs_result,
                reconciled_result=reconciled_result,
                diagnostics_result=diagnostics_result,
                logger=logger,
            ),
        )
    else:
        logger.info(
            "SKIP — Trade Sync decision report is disabled."
        )
        report_result = {
            "status": "disabled",
            "decision_report_path": None,
        }

    workflow_elapsed = perf_counter() - workflow_start

    ready_for_weight_sync = (
        audit_passed
        and reconciled_path.is_file()
        and reconciled_path.stat().st_size > 0
    )

    if not ready_for_weight_sync:
        raise RuntimeError(
            "Trade Sync finished without a valid reconciled-trade output."
        )

    separator(logger)
    logger.info(
        "%s Trade Sync workflow finished.",
        ok,
    )
    logger.info(
        "Final reconciled-trade output: %s",
        reconciled_path,
    )
    logger.info(
        "Final reconciliation audit: %s",
        audit_path,
    )
    logger.info(
        "Ready for Weight Sync: %s",
        ready_for_weight_sync,
    )
    logger.info(
        "Workflow duration: %.2fs",
        workflow_elapsed,
    )

    decision_report_path = report_result.get(
        "decision_report_path"
    )

    if decision_report_path:
        logger.info(
            "Decision report: %s",
            decision_report_path,
        )

    separator(logger)

    return {
        "status": "completed",
        "run_id": run_id,
        "audit_passed": audit_passed,
        "ready_for_weight_sync": ready_for_weight_sync,
        "output_path": reconciled_path,
        "reconciled_trade_path": reconciled_path,
        "audit_path": audit_path,
        "reconciliation_audit_path": audit_path,
        "prepared_input_path": prepared_path,
        "bilateral_pairs_path": pairs_path,
        "diagnostics_path": diagnostics_result.get(
            "diagnostics_path"
        ),
        "decision_report_path": decision_report_path,
        "n_failed_audit_rows": reconciled_result.get(
            "n_failed_audit_rows",
            0,
        ),
        "prepared_result": prepared_result,
        "pairs_result": pairs_result,
        "reconciled_result": reconciled_result,
        "diagnostics_result": diagnostics_result,
        "report_result": report_result,
    }
