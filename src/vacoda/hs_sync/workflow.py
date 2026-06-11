"""
vacoda.hs_sync.workflow
Defines the main workflow orchestration for HS-sync allocation and application.

"""
from __future__ import annotations

from time import perf_counter
from logging import Logger
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
from vacoda.utils.timing import (
    timed_step,
    timed_workflow,
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
]
TOTAL_WORKFLOW_STEPS = len(WORKFLOW_STEPS)


def _infer_step_status(result: dict[str, object]) -> str:
    """
    Infer pass/fail/skipped from a step output dictionary.
    """
    explicit = result.get("status")
    if explicit in {"pass", "fail", "skipped", "disabled", "completed"}:
        if explicit == "completed":
            return "pass"
        if explicit == "disabled":
            return "skipped"
        return str(explicit)

    failed = result.get("n_failed_audit_rows")
    if failed is not None:
        return "pass" if int(failed) == 0 else "fail"

    return "pass"


def _run_step(
    *,
    step_name: str,
    func: Callable[[], dict[str, object]],
    logger: Logger) -> dict[str, object]:
    """
    Run one workflow step and normalize its result.

    The backend function may return any useful dictionary, but this wrapper
    guarantees:
      - status
      - step
      - elapsed_seconds
    """
    t0 = perf_counter()

    try:
        result = func()
    except Exception as exc:
        elapsed = perf_counter() - t0
        logger.exception("Step failed: %s", step_name)

        return {
            "step": step_name,
            "status": "fail",
            "elapsed_seconds": elapsed,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    elapsed = perf_counter() - t0

    if result is None:
        result = {}

    status = _infer_step_status(result)

    return {
        "step": step_name,
        "status": status,
        "elapsed_seconds": elapsed,
        **result,
    }


PASS_STATUSES = {"pass", "completed", "success"}

def _require_pass(
    *,
    outputs: dict[str, object],
    dependency: str,
) -> None:
    dep = outputs.get(dependency)

    if not isinstance(dep, dict):
        raise RuntimeError(f"Missing dependency output: {dependency}")

    status = str(dep.get("status", "")).lower()

    if status not in PASS_STATUSES:
        raise RuntimeError(
            f"Cannot continue: dependency {dependency!r} has status {status!r}."
        )


def _build_workflow_summary(outputs: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for key, value in outputs.items():
        if not isinstance(value, dict):
            continue

        if "step" not in value and "status" not in value:
            continue

        rows.append(
            {
                "name": key,
                "step": value.get("step", key),
                "status": value.get("status", "unknown"),
                "elapsed_seconds": value.get("elapsed_seconds"),
                "n_failed_audit_rows": value.get("n_failed_audit_rows"),
            }
        )

    return rows


def run_hs_sync_workflow(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    logger: Logger,
) -> dict[str, object]:
    outputs: dict[str, object] = {
        "run_id": run_id,
        "status": "started",
    }

    with timed_workflow(logger, "HS-sync workflow"):

        with timed_step(logger, 1, WORKFLOW_STEPS[0], total_steps=TOTAL_WORKFLOW_STEPS):
            outputs["mapping_preparation"] = _run_step(
                step_name=WORKFLOW_STEPS[0],
                logger=logger,
                func=lambda: run_comtrade_mapping_preparation(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )

        with timed_step(logger, 2, WORKFLOW_STEPS[1], total_steps=TOTAL_WORKFLOW_STEPS):
            _require_pass(outputs=outputs, dependency="mapping_preparation")

            coverage_inputs = load_target_hs_coverage_inputs(
                cfg=cfg,
                logger=logger,
            )

            outputs["coverage_inputs"] = {
                "step": WORKFLOW_STEPS[1],
                "status": "pass",
                "input_dir": coverage_inputs["input_dir"],
                "mapping_dir": coverage_inputs["mapping_dir"],
                "output_dir": coverage_inputs["output_dir"],
            }

        with timed_step(logger, 3, WORKFLOW_STEPS[2], total_steps=TOTAL_WORKFLOW_STEPS):
            _require_pass(outputs=outputs, dependency="coverage_inputs")

            coverage_tables = build_target_hs_coverage_products(
                cfg=cfg,
                source_summary=coverage_inputs["source_summary"],
                mappings=coverage_inputs["mappings"],
                logger=logger,
            )

            outputs["coverage_tables"] = {
                "step": WORKFLOW_STEPS[2],
                "status": "pass",
                "tables": {
                    name: table.shape
                    for name, table in coverage_tables.items()
                },
            }

        with timed_step(logger, 4, WORKFLOW_STEPS[3], total_steps=TOTAL_WORKFLOW_STEPS):
            _require_pass(outputs=outputs, dependency="coverage_tables")

            outputs["coverage"] = _run_step(
                step_name=WORKFLOW_STEPS[3],
                logger=logger,
                func=lambda: write_target_hs_coverage_outputs(
                    cfg=cfg,
                    tables=coverage_tables,
                    run_id=run_id,
                    logger=logger,
                ),
            )

        with timed_step(logger, 5, WORKFLOW_STEPS[4], total_steps=TOTAL_WORKFLOW_STEPS):
            _require_pass(outputs=outputs, dependency="coverage")

            outputs["mapping_audit"] = _run_step(
                step_name=WORKFLOW_STEPS[4],
                logger=logger,
                func=lambda: run_comtrade_mapping_audit(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                    expect_no_missing=True,
                    fail_on_error=True,
                ),
            )

        with timed_step(logger, 6, WORKFLOW_STEPS[5], total_steps=TOTAL_WORKFLOW_STEPS):
            _require_pass(outputs=outputs, dependency="mapping_audit")

            outputs["equal_split_allocation"] = _run_step(
                step_name=WORKFLOW_STEPS[5],
                logger=logger,
                func=lambda: run_equal_split_allocation(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )

        with timed_step(logger, 7, WORKFLOW_STEPS[6], total_steps=TOTAL_WORKFLOW_STEPS):
            _require_pass(outputs=outputs, dependency="equal_split_allocation")

            outputs["complete_mapping_weights"] = _run_step(
                step_name=WORKFLOW_STEPS[6],
                logger=logger,
                func=lambda: run_complete_mapping_weights(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )

        with timed_step(logger, 8, WORKFLOW_STEPS[7], total_steps=TOTAL_WORKFLOW_STEPS):
            _require_pass(outputs=outputs, dependency="complete_mapping_weights")

            outputs["apply"] = _run_step(
                step_name=WORKFLOW_STEPS[7],
                logger=logger,
                func=lambda: run_hs_sync_apply(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )

        with timed_step(logger, 9, WORKFLOW_STEPS[8], total_steps=TOTAL_WORKFLOW_STEPS):
            _require_pass(outputs=outputs, dependency="apply")

            outputs["applied_output_audit"] = _run_step(
                step_name=WORKFLOW_STEPS[8],
                logger=logger,
                func=lambda: run_hs_sync_applied_output_audit(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                ),
            )

    apply_outputs = outputs.get("apply", {})
    audit_outputs = outputs.get("applied_output_audit", {})

    if isinstance(apply_outputs, dict):
        outputs["final_parquet"] = apply_outputs.get("output_path")

    if isinstance(audit_outputs, dict):
        outputs["final_audit"] = audit_outputs.get("audit_path")

    outputs["workflow_summary"] = _build_workflow_summary(outputs)

    failed_steps = [
        row for row in outputs["workflow_summary"]
        if row.get("status") == "fail"
    ]

    final_status = "failed" if failed_steps else "completed"
    outputs["status"] = final_status

    logger.info("")
    logger.info("HS-sync workflow step summary:")
    for row in outputs["workflow_summary"]:
        logger.info(
            "  %-30s status=%-8s failed=%s",
            row["name"],
            row["status"],
            row.get("n_failed_audit_rows"),
        )

    outputs["ready_for_next_step"] = final_status == "completed"
    outputs["ready_for_cif_fob"] = final_status == "completed"

    return outputs