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
    target_hs_lower = target_hs.lower()

    output_dir = cfg.weights_output_dir()

    source_code_coverage_path = cfg.weights_input_coverage_path()

    weights_product = cfg.weights.products.get("allocation_weights")
    if weights_product is None:
        raise KeyError("Missing required allocation product: 'allocation_weights'.")

    allocation_weights_path = output_dir / f"{weights_product.filename}.csv"

    if not source_code_coverage_path.exists():
        raise FileNotFoundError(
            f"Missing source-code coverage file: {source_code_coverage_path}"
        )

    if not allocation_weights_path.exists():
        raise FileNotFoundError(
            f"Missing equal-split allocation weights file: {allocation_weights_path}"
        )

    # If you later add allocation.products.complete_mapping_weights to YAML,
    # use that. For now, keep the already proven filename prefix.
    complete_product = cfg.weights.products.get("complete_mapping_weights")
    if complete_product is not None:
        suffix = "_complete_mapping_weights"
        if not complete_product.filename.endswith(suffix):
            raise ValueError(
                "allocation.products.complete_mapping_weights.filename must end "
                f"with {suffix!r}; got {complete_product.filename!r}."
            )
        filename_prefix = complete_product.filename[: -len(suffix)]
    else:
        filename_prefix = f"target_{target_hs_lower}"

    logger.info("Starting complete target-HS mapping-weight step.")
    logger.info("  Run ID: %s", run_id)
    logger.info("  Target HS: %s", target_hs)
    logger.info("  Source coverage: %s", source_code_coverage_path)
    logger.info("  Allocation weights: %s", allocation_weights_path)
    logger.info("  Output directory: %s", output_dir)
    logger.info("  Output filename prefix: %s", filename_prefix)

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
        output_dir=output_dir,
        filename_prefix=filename_prefix,
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