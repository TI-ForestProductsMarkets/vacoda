# scripts/hs_sync/run_hs_sync_coverage_flow.py

"""
Run the VACODA HS-sync coverage/audit flow.

This script orchestrates:

1. Harvard-compatible Comtrade staging.
2. Harvard-generator input audit.
3. Harvard mapping-direction audit.
4. Target-HS mapping coverage audit.

It does not apply weighted HS harmonization.
"""

from __future__ import annotations

from datetime import datetime
from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.pipeline import PipelineConfig
from vacoda.config.paths import resolve_log_path
from vacoda.hs_sync.harvard.coverage_flow import run_hs_sync_coverage_flow
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)


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

    logger = get_logger("SCRIPT.HS_SYNC_COVERAGE_FLOW")

    try:
        outputs = run_hs_sync_coverage_flow(
            cfg=cfg,
            run_id=run_id,
        )
    except Exception:
        logger.exception("HS-sync coverage flow failed.")
        logger.info("Log file: %s", log_path)
        raise

    logger.info("")
    logger.info("HS-sync coverage flow finished.")
    logger.info("  Status: %s", outputs.get("status"))
    logger.info("  Run ID: %s", outputs.get("run_id"))
    logger.info("  Log file: %s", log_path)

    coverage_outputs = outputs.get("target_hs_mapping_coverage", {})
    logger.info(
        "  Final coverage artifact: %s",
        coverage_outputs.get("coverage_report_latest"),
    )


if __name__ == "__main__":
    main()