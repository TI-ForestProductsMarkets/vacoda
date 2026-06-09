"""
vacoda.hs_sync.weights
Functions for building the complete target-HS mapping-weight table, which is the main input to the HS-sync apply step.

"""
from __future__ import annotations

from vacoda.config.hs_sync import HsSyncConfig
import pandas as pd
from vacoda.hs_sync.allocation import build_complete_mapping_weight_table, write_complete_mapping_weight_outputs

def run_complete_mapping_weights(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    logger: logger.Logger,
) -> dict[str, object]:
    """
    Build the complete target-HS mapping-weight table.

    This combines:
    - deterministic/easy mappings with allocation_weight = 1.0
    - non-deterministic mappings with equal-split allocation weights

    The result is the single source -> target mapping-weight table used by
    the later HS-sync apply step.
    """

    if not cfg.weights.enabled:
        logger.info("Allocation is disabled in config; skipping complete weights.")
        return {
            "status": "disabled",
            "run_id": run_id,
        }

    target_hs = cfg.weights.target_hs

    source_code_coverage_path = cfg.weights_input_coverage_path()
    allocation_weights_path = cfg.weights_equal_split_weights_path()
    output_weights_path = cfg.weights_complete_mapping_weights_path()
    output_audit_path = cfg.weights_complete_mapping_audit_path()

    if not source_code_coverage_path.exists():
        raise FileNotFoundError(
            f"Missing source-code coverage file: {source_code_coverage_path}"
        )

    if not allocation_weights_path.exists():
        raise FileNotFoundError(
            f"Missing equal-split allocation weights file: {allocation_weights_path}"
        )

    logger.info("Starting complete target-HS mapping-weight step.")
    logger.info("  Run ID: %s", run_id)
    logger.info("  Target HS: %s", target_hs)
    logger.info("  Source coverage: %s", source_code_coverage_path)
    logger.info("  Allocation weights: %s", allocation_weights_path)
    logger.info("  Output weights path: %s", output_weights_path)
    logger.info("  Output audit path: %s", output_audit_path)

    source_code_coverage = pd.read_csv(
        source_code_coverage_path,
        dtype={
            "source_cmd": str,
            "target_codes": str,
        },
    )

    allocation_weights = pd.read_csv(
        allocation_weights_path,
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

    n_failed = int((result.audit["status"] != "pass").sum())

    logger.info("Complete mapping-weight step completed.")
    logger.info("  Complete weight rows: %s", f"{len(result.weights):,}")
    logger.info("  Complete audit rows: %s", f"{len(result.audit):,}")
    logger.info("  Failed audit rows: %s", f"{n_failed:,}")
    logger.info("  Complete weights file: %s", outputs["complete_weights"])
    logger.info("  Complete weights audit file: %s", outputs["complete_weights_audit"])

    return {
        "status": "completed",
        "run_id": run_id,
        "complete_weights": outputs["complete_weights"],
        "complete_weights_audit": outputs["complete_weights_audit"],
        "n_weight_rows": len(result.weights),
        "n_audit_rows": len(result.audit),
        "n_failed_audit_rows": n_failed,
    }