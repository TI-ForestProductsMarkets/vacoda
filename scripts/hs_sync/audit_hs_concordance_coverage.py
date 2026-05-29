"""
scripts.hs_sync.audit_hs_concordance_coverage
Script to audit HS concordance coverage.

This script checks whether the HS systems and commodity codes present in the
source trade data can be mapped to the configured target HS system using the
configured concordance table.

It produces diagnostic tables such as:
- HS concordance coverage
- missing HS mappings
- one-to-many HS mappings
- value-split HS mappings

This is a diagnostic step before applying weighted HS harmonization.
"""

from __future__ import annotations

from datetime import datetime

from vacoda.config.hs_sync import (
    get_logging_file,
    get_logging_level,
    get_logging_timestamp_bool,
    load_hs_sync_config,
)
from vacoda.config.paths import resolve_log_path
from vacoda.hs_sync.workflow import run_hs_sync_diagnostics
from vacoda.utils.logger import get_logger, parse_log_level, setup_logging


def main() -> None:
    cfg = load_hs_sync_config()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(get_logging_level(cfg))

    log_path = resolve_log_path(
        get_logging_file(cfg),
        timestamp=run_id if get_logging_timestamp_bool(cfg) else None,
    )

    setup_logging(level=log_level, log_file=log_path)

    logger = get_logger("SCRIPT.HS_SYNC")
    logger.info("Starting HS-sync concordance coverage audit.")
    logger.debug("Log file: %s", log_path)

    try:
        outputs = run_hs_sync_diagnostics(run_id=run_id)
    except Exception:
        logger.exception("HS-sync concordance coverage audit failed.")
        logger.info("Log file: %s", log_path)
        raise

    logger.info("Finished HS-sync concordance coverage audit.")
    logger.info("")
    logger.info("HS-sync diagnostics workflow complete.")
    logger.info("  Output directory: %s", outputs.get("output_dir"))
    logger.info("  Log file: %s", log_path)

    for name, path in outputs.items():
        if name == "output_dir":
            continue
        logger.info("  %s: %s", name, path)


if __name__ == "__main__":
    main()