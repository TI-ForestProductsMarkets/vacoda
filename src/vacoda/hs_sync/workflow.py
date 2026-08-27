"""
vacoda.hs_sync.workflow
Defines the main workflow orchestration for HS-sync allocation and application.

"""
from __future__ import annotations
from time import perf_counter

from logging import Logger
from vacoda.config.hs_sync import HsSyncConfig
from vacoda.hs_sync.mapping import run_comtrade_mapping_preparation
from vacoda.hs_sync.coverage_audit import (
    load_target_hs_coverage_inputs,
    write_target_hs_coverage_outputs,
    build_target_hs_coverage_products,
)
from vacoda.hs_sync.mapping_audit import run_comtrade_mapping_audit
from vacoda.hs_sync.allocation import run_equal_split_allocation
from vacoda.hs_sync.weights import run_complete_mapping_weights
from vacoda.hs_sync.apply import run_hs_sync_apply
from vacoda.hs_sync.applied_audit import run_hs_sync_applied_output_audit
from vacoda.hs_sync.reporting import build_hs_sync_report
from vacoda.utils.runtime import (
    timed_workflow,
    run_step,
)
from vacoda.utils.artifacts import require_artifact, require_enabled_products

REQUIRED_PRODUCTS = (
    "coverage",
    "equal_split_allocation",
    "complete_mapping_weights",
    "hs_sync_apply",
)

WORKFLOW_STEPS = [
    "Prepare UNSD/Comtrade HS mappings",
    "Load coverage inputs",
    "Build coverage products",
    "Write coverage outputs",
    "Audit Comtrade mapping normalization",
    "Build equal-split allocation weights",
    "Build complete HS mapping weights",
    "Apply HS-sync to target HS",
    "Audit applied HS-sync output",
    "Write HS-sync decision report",
]
TOTAL_WORKFLOW_STEPS = len(WORKFLOW_STEPS)


def load_coverage_inputs_step(
    *,
    cfg: HsSyncConfig,
    logger: Logger,
) -> dict[str, object]:
    coverage_inputs = load_target_hs_coverage_inputs(
        cfg=cfg,
        logger=logger,
    )

    return {
        "status": "completed",
        "coverage_inputs": coverage_inputs,
        "input_dir": coverage_inputs["input_dir"],
        "mapping_dir": coverage_inputs["mapping_dir"],
        "output_dir": coverage_inputs["output_dir"],
    }

def build_coverage_tables_step(
    *,
    cfg: HsSyncConfig,
    coverage_inputs: dict[str, object],
    logger: Logger,
) -> dict[str, object]:
    tables = build_target_hs_coverage_products(
        cfg=cfg,
        source_summary=coverage_inputs["source_summary"],
        mappings=coverage_inputs["mappings"],
        logger=logger,
    )

    return {
        "status": "completed",
        "coverage_tables": tables,
        "table_shapes": {
            name: table.shape
            for name, table in tables.items()
        },
    }


def run_hs_sync_workflow(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    logger: Logger,
    force: bool = False,
) -> dict[str, object]:

    """
    Run the HS-sync workflow, which includes the following steps:
    
    Parameters:
        cfg:
            (HsSyncConfig): Configuration for the HS-sync workflow.
        run_id (str): 
            Unique identifier for this workflow run.
        logger (Logger): 
            Logger for logging workflow progress and information.
        force_run (bool): 
            Whether to force the workflow to run, ignoring cached results.
        input_path (Path | None):
            Optional path to input data for the workflow. When omitted, ``cfg.input_dir`` is used.

        Expected step modules:
            - run_comtrade_mapping_preparation
            - load_target_hs_coverage_inputs
            - build_target_hs_coverage_products
            - write_target_hs_coverage_outputs
            - run_comtrade_mapping_audit
            - run_equal_split_allocation
            - run_complete_mapping_weights
            - run_hs_sync_apply
            - run_hs_sync_audit
    """

    input_dir = cfg.coverage_input_dir()
    input_pattern = cfg.coverage.input.pattern

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"HS Sync input directory not found: {input_dir}"
        )

    if not input_pattern:
        raise ValueError(
            "HS Sync coverage input pattern is not configured."
        )

    input_files = sorted(input_dir.glob(input_pattern))

    if not input_files:
        raise FileNotFoundError(
            "No HS Sync input files matched "
            f"{input_dir / input_pattern}"
        )

    enabled_products = cfg.workflow_enabled_products()

    require_enabled_products(
        workflow_name="HS Sync",
        enabled_products=enabled_products,
        required_products=REQUIRED_PRODUCTS,
    )

    workflow_start = perf_counter()

    with timed_workflow(logger, "HS-sync workflow"):

        logger.info("Run ID: %s", run_id)
        logger.info("Target HS: %s", cfg.scope.target_hs)
        logger.info(
            "Year range: %s-%s",
            cfg.scope.min_year,
            cfg.scope.max_year,
        )
        logger.info("Input: %s", input_dir)
        logger.info("Input directory: %s", input_dir)
        logger.info("Input pattern: %s", input_pattern)
        logger.info("Matched input files: %s", len(input_files))
        logger.info(
            "Enabled products: %s",
            " ".join(
                f"{name}={enabled}"
                for name, enabled in enabled_products.items()
            ),
        )

        # ==================================================================
        # Step 1 —  Prepare UNSD/Comtrade HS mappings
        # ==================================================================

        mapping_result = run_step(
                logger=logger,
                step_number=1,
                total_steps=TOTAL_WORKFLOW_STEPS,
                title=WORKFLOW_STEPS[0],
                runner=lambda: run_comtrade_mapping_preparation(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )

        # ==================================================================
        # Step 2 — Build coverage products
        # ==================================================================

        coverage_inputs_result = run_step(
            logger=logger,
            step_number=2,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[1],
            runner=lambda: load_coverage_inputs_step(
                cfg=cfg,
                logger=logger,
            ),
        )

        coverage_inputs = coverage_inputs_result["coverage_inputs"]

        # ==================================================================
        # Step 3 — Build coverage tables
        # ==================================================================

        coverage_tables_result = run_step(
            logger=logger,
            step_number=3,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[2],
            runner=lambda: build_coverage_tables_step(
                cfg=cfg,
                coverage_inputs=coverage_inputs,
                logger=logger,
            ),
        )

        coverage_tables = coverage_tables_result["coverage_tables"]


        # ==================================================================
        # Step 4 — Build coverage outputs
        # ==================================================================

        coverage_result  = run_step(
            logger=logger,
            step_number=4,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[3],
            runner=lambda: write_target_hs_coverage_outputs(
                    cfg=cfg,
                    tables=coverage_tables,
                    run_id=run_id,
                    logger=logger,
                ),
            )

        # ==================================================================
        # Step 5 — Run mapping audit
        # ==================================================================

        mapping_audit_result  = run_step(
            step_number=5,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[4],
            logger=logger,
            runner=lambda: run_comtrade_mapping_audit(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                    expect_no_missing=True,
                    fail_on_error=True,
                ),
            )

        # ==================================================================
        # Step 6 — Run equal split allocation
        # ==================================================================

        equal_split_result  = run_step(
            step_number=6,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[5],
            logger=logger,
            runner=lambda: run_equal_split_allocation(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )

        # ==================================================================
        # Step 7 — Run complete mapping weights
        # ==================================================================

        complete_weights_result  = run_step(
            step_number=7,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[6],
            logger=logger,
            runner=lambda: run_complete_mapping_weights(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )

        # ==================================================================
        # Step 8 — Run HS sync apply
        # ==================================================================

        apply_result  = run_step(
            step_number=8,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[7],
            logger=logger,
            runner=lambda: run_hs_sync_apply(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )
        applied_path = require_artifact(
                    result=apply_result,
                    path_key="output_path",
                    consumer_step=WORKFLOW_STEPS[8],
                    producer_product="HS-synced trade Parquet",
                )

        # ==================================================================
        # Step 9 — Run HS sync applied output audit
        # ==================================================================

        applied_audit_result = run_step(
            step_number=9,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[8],
            logger=logger,
            runner=lambda: run_hs_sync_applied_output_audit(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )
        
        audit_path = require_artifact(
                result=applied_audit_result,
                path_key="audit_path",
                consumer_step="HS Sync readiness validation",
                producer_product="applied HS Sync audit",
            )

        ready_for_quantity = (
            applied_audit_result.get("ready_for_quantity") is True
        )

        if not ready_for_quantity:
            raise RuntimeError(
                "HS Sync applied-output audit did not approve "
                "the output for quantity."
            )

        n_failed_audit_rows = int(
                    applied_audit_result.get("n_failed_audit_rows", 0)
                )

        # ==================================================================
        # Step 10 — Write HS-sync decision report
        # ==================================================================
        decision_report_result = run_step(
            logger=logger,
            step_number=10,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[9],
            runner=lambda: build_hs_sync_report(
                    cfg=cfg,
                    tables={
                        "hs_sync_decision_report": applied_audit_result.get(
                            "decision_report"
                        )
                    },
                    run_id=run_id,
                    force=force,
                    logger=logger,
                ),
            )



       ########################################################################################
        
        workflow_elapsed = perf_counter() - workflow_start

        logger.info("HS Sync workflow finished.")
        logger.info("Applied output: %s", applied_path)
        logger.info("Applied-output audit: %s", audit_path)
        logger.info("Audit passed: %s", ready_for_quantity)
        logger.info("Failed audit rows: %s", n_failed_audit_rows)
        logger.info("Ready for quantity: %s", ready_for_quantity)
        logger.info("Workflow elapsed time: %.2f seconds", workflow_elapsed)

    return {
            "status": "completed",
            "run_id": run_id,

            # Master-pipeline readiness contract.
            "ready_for_quantity": ready_for_quantity,

            # Compatibility alias.
            "ready_for_next_step": ready_for_quantity,

            # Canonical primary output and audit keys used by the master
            # pipeline and checkpoint system.
            "output_path": applied_path,
            "audit_path": audit_path,

            # Explicit HS Sync product paths.
            "final_parquet": applied_path,
            "final_audit": audit_path,

            # Final validation outcome.
            "audit_passed": ready_for_quantity,
            "n_failed_audit_rows": n_failed_audit_rows,

            # Useful final workflow metadata.
            "workflow_duration_seconds": workflow_elapsed,
        }