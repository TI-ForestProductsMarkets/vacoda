"""
Apply HS-sync mapping weights to Comtrade rows.

This converts observed trade rows from their reported HS system into the
configured target HS system, currently H5/HS2017.

It uses the complete mapping-weight table produced by:

    scripts/hs_sync/03_build_complete_h5_mapping_weights.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.paths import resolve_log_path
from vacoda.hs_sync.apply import apply_hs_sync_with_weights
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)


logger = get_logger("SCRIPT.HS_SYNC_APPLY")


def main() -> None:
    cfg = HsSyncConfig.from_yaml()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(cfg.logging.level)
    log_path = resolve_log_path(
        cfg.logging.resolve_log_file(),
        timestamp=cfg.logging.timestamp,
    )

    setup_logging(level=log_level, log_file=log_path)

    target_hs = cfg.coverage.target_hs
    target_hs_lower = target_hs.lower()

    input_dir = cfg.coverage_input_dir()
    input_glob = str(input_dir / cfg.coverage.input.pattern)

    allocation_dir = Path(
        f"/storage/vacoda/data/interim/hs_sync/target_{target_hs_lower}_allocation"
    )

    weights_path = (
        allocation_dir / f"target_{target_hs_lower}_complete_mapping_weights.csv"
    )

    output_dir = Path(
        f"/storage/vacoda/data/interim/hs_sync/target_{target_hs_lower}_applied"
    )

    output_path = output_dir / f"comtrade_target_{target_hs_lower}_hs_synced.parquet"
    audit_path = output_dir / f"comtrade_target_{target_hs_lower}_hs_sync_audit.csv"

    logger.info("Starting HS-sync apply workflow.")
    logger.info("  Target HS: %s", target_hs)
    logger.info("  Input glob: %s", input_glob)
    logger.info("  Mapping weights: %s", weights_path)
    logger.info("  Output path: %s", output_path)
    logger.info("  Audit path: %s", audit_path)

    result = apply_hs_sync_with_weights(
        input_glob=input_glob,
        weights_path=weights_path,
        output_path=output_path,
        audit_path=audit_path,
        min_year=cfg.coverage.min_year,
        max_year=cfg.coverage.max_year,
        year_col=cfg.columns.year,
        hs_col=cfg.columns.hs,
        cmd_col=cfg.columns.cmd,
        value_col=cfg.columns.value,
        output_format="parquet",
        threads=cfg.engine.threads,
        overwrite=True,
    )

    logger.info("HS-sync apply workflow completed.")
    logger.info("  Output file: %s", result.output_path)
    logger.info("  Audit file: %s", result.audit_path)
    logger.info("  Log file: %s", log_path)


if __name__ == "__main__":
    main()