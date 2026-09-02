"""
vacoda.value.workflow

Orchestrate value deflation after Trade Sync.

This module intentionally contains orchestration only. The scientific
logic belongs in the individual Value step modules.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from vacoda.config.value import ValueConfig
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

from vacoda.value.prepare_deflator import prepare_us_gdp_deflator
from vacoda.value.deflate import deflate_trade_values
from vacoda.value.audit import run_value_output_audit
from vacoda.value.reporting import build_value_report

# These will be added as the module is built:
# from vacoda.value.deflate import deflate_trade_values
# from vacoda.value.audit import run_value_output_audit
# from vacoda.value.reporting import build_value_report


REQUIRED_PRODUCTS = (
    "deflated_trade",
    "audit",
)


WORKFLOW_STEPS = [
    "Prepare US GDP deflator",
    "Deflate synchronized trade values",
    "Audit Value output",
    "Write Value decision report",
]


TOTAL_WORKFLOW_STEPS = len(WORKFLOW_STEPS)


def run_value_workflow(
    *,
    cfg: ValueConfig,
    run_id: str,
    logger,
    force: bool = False,
    input_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run the complete Value workflow.

    Parameters
    ----------
    cfg:
        Parsed Value configuration.
    run_id:
        Master-pipeline run identifier.
    logger:
        Configured workflow logger.
    force:
        Ignore reusable artifacts inside Value step modules.
    input_path:
        Optional explicit Trade-Sync input. When omitted,
        ``cfg.input_path()`` is used.
    """

    resolved_input_path = (
        Path(input_path)
        if input_path is not None
        else cfg.input_path()
    )

    if not resolved_input_path.is_file():
        raise FileNotFoundError(
            "Value input parquet not found: "
            f"{resolved_input_path}"
        )

    enabled_products = cfg.enabled_products()

    require_enabled_products(
        workflow_name="Value",
        enabled_products=enabled_products,
        required_products=REQUIRED_PRODUCTS,
    )

    require_product_dependency(
        workflow_name="Value",
        enabled_products=enabled_products,
        product="decision_report",
        required_product="audit",
    )

    cfg.output_dir().mkdir(
        parents=True,
        exist_ok=True,
    )

    workflow_start = perf_counter()

    with timed_workflow(logger, "Value"):

        logger.info("Run ID: %s", run_id)
        logger.info("Target HS: %s", cfg.scope.target_hs)
        logger.info(
            "Year range: %s-%s",
            cfg.scope.min_year,
            cfg.scope.max_year,
        )
        logger.info(
            "Input: %s",
            resolved_input_path,
        )
        logger.info(
            "Output directory: %s",
            cfg.output_dir(),
        )
        logger.info(
            "GDP deflator source: %s",
            cfg.deflator_source_path(),
        )
        logger.info(
            "Prepared GDP deflator: %s",
            cfg.deflator_prepared_path(),
        )
        logger.info(
            "Reference year: %s",
            cfg.method.reference_year,
        )
        logger.info(
            "Preserve nominal values: %s",
            cfg.method.preserve_nominal_values,
        )
        logger.info(
            "Enabled products: %s",
            " ".join(
                f"{name}={enabled}"
                for name, enabled in enabled_products.items()
            ),
        )

        # ==================================================================
        # Step 1 — Prepare canonical GDP-deflator reference
        # ==================================================================

        deflator_result = run_step(
            logger=logger,
            step_number=1,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[0],
            runner=lambda: prepare_us_gdp_deflator(
                cfg=cfg,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        deflator_path = require_artifact(
            result=deflator_result,
            path_key="prepared_deflator_path",
            consumer_step=WORKFLOW_STEPS[1],
            producer_product="prepared US GDP-deflator Parquet",
        )

        # ==================================================================
        # Step 2 — Deflate synchronized trade values
        # ==================================================================

        deflated_result = run_step(
            logger=logger,
            step_number=2,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[1],
            runner=lambda: deflate_trade_values(
                cfg=cfg,
                input_path=resolved_input_path,
                deflator_path=deflator_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        deflated_path = require_artifact(
            result=deflated_result,
            path_key="deflated_trade_path",
            consumer_step=WORKFLOW_STEPS[2],
            producer_product="deflated Value trade Parquet",
        )

        # ==================================================================
        # Step 3 — Audit Value output
        # ==================================================================

        audit_result = run_step(
            logger=logger,
            step_number=3,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[2],
            runner=lambda: run_value_output_audit(
                cfg=cfg,
                input_path=resolved_input_path,
                deflated_trade_path=deflated_path,
                deflator_path=deflator_path,
                run_id=run_id,
                logger=logger,
            ),
        )

        audit_path = require_artifact(
            result=audit_result,
            path_key="audit_path",
            consumer_step=WORKFLOW_STEPS[3],
            producer_product="Value output audit",
        )

        # ==================================================================
        # Step 4 — Write Value decision report
        # ==================================================================

        report_result = run_step(
            logger=logger,
            step_number=4,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[3],
            runner=lambda: build_value_report(
                cfg=cfg,
                input_path=resolved_input_path,
                deflated_trade_path=deflated_path,
                deflator_path=deflator_path,
                audit_path=audit_path,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )

        report_path = require_result_path(
            report_result,
            "Value reporting result did not contain "
            "an expected report path.",
            "report_path",
            "output_path",
        )

        if not report_path.is_file():
            raise FileNotFoundError(
                f"Value decision report not found: {report_path}"
            )

        ready_for_validation = bool(
            report_result.get(
                "ready_for_validation"
            )
        )

        if not ready_for_validation:
            raise RuntimeError(
                "Value workflow finished without "
                "validation-ready output."
            )

        workflow_elapsed = (
            perf_counter()
            - workflow_start
        )

        separator(logger)

        logger.info(
            "%s Value workflow finished.",
            get_symbols("success"),
        )
        logger.info(
            "Run ID: %s",
            run_id,
        )
        logger.info(
            "Prepared GDP deflator: %s",
            deflator_path,
        )
        logger.info(
            "Deflated trade: %s",
            deflated_path,
        )
        logger.info(
            "Audit: %s",
            audit_path,
        )
        logger.info(
            "Decision report: %s",
            report_path,
        )

        latest_report_path = (
            report_result.get(
                "latest_report_path"
            )
        )

        if latest_report_path is not None:
            logger.info(
                "Latest decision report: %s",
                latest_report_path,
            )

        logger.info(
            "Ready for validation: %s",
            ready_for_validation,
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
        "ready_for_validation": (
            ready_for_validation
        ),

        # Canonical primary output.
        "output_path": deflated_path,

        # Explicit Value product paths.
        "prepared_deflator_path": (
            deflator_path
        ),
        "deflated_trade_path": (
            deflated_path
        ),
        "audit_path": audit_path,
        "report_path": report_path,
        "latest_report_path": (
            report_result.get(
                "latest_report_path"
            )
        ),

        # Final decision metadata.
        "decision": (
            report_result.get(
                "decision"
            )
        ),
        "n_failed_audit_rows": (
            report_result.get(
                "n_failed_audit_rows"
            )
        ),
        "workflow_duration_seconds": (
            workflow_elapsed
        ),
    }