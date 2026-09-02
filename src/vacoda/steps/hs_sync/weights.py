"""
vacoda.hs_sync.weights
Functions for building the complete target-HS mapping-weight table, which is the main input to the HS-sync apply step.

"""
from __future__ import annotations

from vacoda.config.steps.hs_sync import HsSyncConfig
import pandas as pd
from vacoda.steps.hs_sync.allocation import build_complete_mapping_weight_table, write_complete_mapping_weight_outputs

def run_complete_mapping_weights(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    force: bool = False,
    logger: None | object = None,
) -> dict[str, object]:

    if not cfg.weights.enabled:
        logger.info(
            "Allocation is disabled in config; skipping complete weights."
        )
        return {
            "status": "disabled",
            "run_id": run_id,
        }

    target_hs = cfg.weights.target_hs

    source_code_coverage_path = (
        cfg.weights_input_coverage_path()
    )
    weights_path = (
        cfg.weights_equal_split_weights_path()
    )
    output_weights_path = (
        cfg.weights_complete_mapping_weights_path()
    )
    output_audit_path = (
        cfg.weights_complete_mapping_audit_path()
    )

    if not source_code_coverage_path.is_file():
        raise FileNotFoundError(
            "Missing source-code coverage file: "
            f"{source_code_coverage_path}"
        )

    if not weights_path.is_file():
        raise FileNotFoundError(
            "Missing equal-split allocation weights file: "
            f"{weights_path}"
        )


    # --------------------------------------------------------------
    # Existing build
    # --------------------------------------------------------------

    logger.info(
        "Starting complete target-HS mapping-weight step."
    )
    logger.info("  Run ID: %s", run_id)
    logger.info("  Target HS: %s", target_hs)
    logger.info(
        "  Source coverage: %s",
        source_code_coverage_path,
    )
    logger.info(
        "  Allocation weights: %s",
        weights_path,
    )
    logger.info(
        "  Output weights path: %s",
        output_weights_path,
    )
    logger.info(
        "  Output audit path: %s",
        output_audit_path,
    )

    source_code_coverage = pd.read_csv(
        source_code_coverage_path,
        dtype={
            "source_cmd": str,
            "target_codes": str,
        },
    )

    allocation_weights = pd.read_csv(
        weights_path,
        dtype={
            "source_cmd": str,
            "target_cmd": str,
        },
    )

    logger.info(
        "Loaded source-code coverage rows: %s",
        f"{len(source_code_coverage):,}",
    )
    logger.info(
        "Loaded allocation weight rows: %s",
        f"{len(allocation_weights):,}",
    )

    result = build_complete_mapping_weight_table(
        source_code_coverage=source_code_coverage,
        allocation_weights=allocation_weights,
    )

    outputs = write_complete_mapping_weight_outputs(
        weights=result.weights,
        audit=result.audit,
        weights_path=output_weights_path,
        audit_path=output_audit_path,
        overwrite=cfg.weights.write.overwrite,
    )

    n_failed = int(
        (result.audit["status"] != "pass").sum()
    )

    audit_passed = n_failed == 0

    if audit_passed:
        logger.info(
            "✅ AUDIT — Complete HS mapping weights passed."
        )

    logger.info(
        "Complete mapping-weight step completed."
    )
    logger.info(
        "  Complete weight rows: %s",
        f"{len(result.weights):,}",
    )
    logger.info(
        "  Complete audit rows: %s",
        f"{len(result.audit):,}",
    )
    logger.info(
        "  Failed audit rows: %s",
        f"{n_failed:,}",
    )
    logger.info(
        "  Complete weights file: %s",
        outputs["complete_weights"],
    )
    logger.info(
        "  Complete weights audit file: %s",
        outputs["complete_weights_audit"],
    )

    return {
        "status": "completed",
        "run_id": run_id,
        "complete_weights": outputs["complete_weights"],
        "complete_weights_audit": (
            outputs["complete_weights_audit"]
        ),
        "audit_passed": audit_passed,
        "n_weight_rows": len(result.weights),
        "n_audit_rows": len(result.audit),
        "n_failed_audit_rows": n_failed,
    }