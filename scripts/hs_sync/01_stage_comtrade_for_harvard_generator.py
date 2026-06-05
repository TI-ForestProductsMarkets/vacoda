"""
scripts.hs_sync.01_stage_comtrade_for_harvard_generator
Read raw Comtrade CSV/GZ data and create Harvard-compatible staged parquet.
Also fixes/exports reporter/partner ISO compatibility if bulk data lacks ISO fields.
"""

from __future__ import annotations

from datetime import datetime

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.paths import resolve_log_path
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)

from vacoda.hs_sync.harvard_stage import run_harvard_staging_pipeline


def main() -> None:
    cfg = HsSyncConfig.from_yaml()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(cfg.logging.level)
    log_path = resolve_log_path(
        cfg.logging.resolve_log_file(),
        timestamp=cfg.logging.timestamp,
    )

    setup_logging(level=log_level, log_file=log_path)

    logger = get_logger("SCRIPT.HS_SYNC_STAGE_HARVARD")

    try:
        outputs = run_harvard_staging_pipeline(
            cfg=cfg,
            run_id=run_id,
        )
    except Exception:
        logger.exception("Harvard staging failed.")
        logger.info("Log file: %s", log_path)
        raise

    logger.info("Harvard staging workflow complete.")
    logger.info("  Status: %s", outputs.get("status", "completed"))
    logger.info("  Output directory: %s", outputs.get("output_dir"))

    logger.info("  Artifact directory: %s", outputs.get("artifact_dir"))

    if outputs.get("summary") is not None:
        logger.info("  Summary: %s", outputs.get("summary"))

    if outputs.get("compatibility_summary") is not None:
        logger.info(
            "  Compatibility summary: %s",
            outputs.get("compatibility_summary"),
        )

    if outputs.get("marker_path") is not None:
        logger.info("  Success marker: %s", outputs.get("marker_path"))

    if outputs.get("skip_reason") is not None:
        logger.info("  Skip reason: %s", outputs.get("skip_reason"))

    logger.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()