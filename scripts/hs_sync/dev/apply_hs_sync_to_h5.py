from __future__ import annotations

from datetime import datetime

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.paths import resolve_log_path
from vacoda.hs_sync.apply import run_hs_sync_apply_from_config
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

    outputs = run_hs_sync_apply_from_config(
        cfg=cfg,
        run_id=run_id,
        logger=logger,
    )

    logger.info("HS-sync apply script finished.")
    logger.info("  Status: %s", outputs.get("status"))
    logger.info("  Output file: %s", outputs.get("output_path"))
    logger.info("  Audit file: %s", outputs.get("audit_path"))
    logger.info("  Log file: %s", log_path)


if __name__ == "__main__":
    main()