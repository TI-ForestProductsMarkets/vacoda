"""
Run VACODA HS-sync workflow.
"""

from __future__ import annotations

from datetime import datetime

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.pipeline import PipelineConfig
from vacoda.config.paths import resolve_log_path
from vacoda.hs_sync.workflow import run_hs_sync_workflow
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

    logger = get_logger("WORKFLOW.HS_SYNC")

    try:
        outputs = run_hs_sync_workflow(
            cfg=cfg,
            run_id=run_id,
            logger=logger,
        )
    except Exception:
        logger.exception("HS-sync workflow failed.")
        logger.info("Log file: %s", log_path)
        raise

    logger.info("HS-sync workflow finished.")
    logger.info("  Status: %s", outputs.get("status"))
    logger.info("  Final parquet: %s", outputs.get("final_parquet"))
    logger.info("  Final audit: %s", outputs.get("final_audit"))
    logger.info("  Log file: %s", log_path)


if __name__ == "__main__":
    main()