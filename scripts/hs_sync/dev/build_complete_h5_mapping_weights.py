"""
Build complete H5 mapping-weight table.

This combines:
- deterministic/easy mappings with allocation_weight = 1.0
- non-deterministic mappings with equal-split allocation weights

The result is the single source→target mapping-weight table used by the
later HS-sync apply step.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from vacoda.config.steps.hs_sync import HsSyncConfig
from vacoda.config.pipeline import PipelineConfig
from vacoda.config.paths import resolve_log_path
from vacoda.steps.hs_sync.allocation import (
    build_complete_mapping_weight_table,
    write_complete_mapping_weight_outputs,
)
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)


logger = get_logger("SCRIPT.HS_SYNC_COMPLETE_WEIGHTS")


def main() -> None:
    pipeline_cfg = PipelineConfig.from_yaml()
    cfg = HsSyncConfig.from_yaml(
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(cfg.logging.level)
    log_path = resolve_log_path(
        cfg.logging.resolve_log_file(),
        timestamp=cfg.logging.timestamp,
    )

    setup_logging(level=log_level, log_file=log_path)

    target_hs = cfg.coverage.target_hs
    target_hs_lower = target_hs.lower()

    coverage_dir = cfg.coverage_output_dir()
    allocation_dir = Path(
        f"/storage/vacoda/data/interim/hs_sync/target_{target_hs_lower}_allocation"
    )

    source_code_coverage_path = (
        coverage_dir / f"target_{target_hs_lower}_source_code_coverage.csv"
    )

    allocation_weights_path = (
        allocation_dir / f"target_{target_hs_lower}_equal_split_allocation_weights.csv"
    )

    output_dir = allocation_dir
    filename_prefix = f"target_{target_hs_lower}"

    logger.info("Starting complete H5 mapping-weight workflow.")
    logger.info("  Target HS: %s", target_hs)
    logger.info("  Source coverage: %s", source_code_coverage_path)
    logger.info("  Allocation weights: %s", allocation_weights_path)
    logger.info("  Output directory: %s", output_dir)

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

    result = build_complete_mapping_weight_table(
        source_code_coverage=source_code_coverage,
        allocation_weights=allocation_weights,
    )

    outputs = write_complete_mapping_weight_outputs(
        weights=result.weights,
        audit=result.audit,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        overwrite=True,
    )

    n_failed = int((result.audit["status"] != "pass").sum())

    logger.info("Complete mapping-weight workflow completed.")
    logger.info("  Complete weight rows: %s", f"{len(result.weights):,}")
    logger.info("  Complete audit rows: %s", f"{len(result.audit):,}")
    logger.info("  Failed audit rows: %s", f"{n_failed:,}")
    logger.info("  Complete weights file: %s", outputs["complete_weights"])
    logger.info("  Complete weights audit file: %s", outputs["complete_weights_audit"])
    logger.info("  Log file: %s", log_path)


if __name__ == "__main__":
    main()