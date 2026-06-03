"""
scripts.hs_sync.05_audit_target_hs_mapping_coverage.py

Script to audit target HS mapping coverage.

Given observed trade data and configured mapping files, classify source HS-code pairs into:
easy_without_weights, needs_weights, missing/unsupported.

It produces diagnostic tables such as:
- mapping coverage
- source-code coverage
- mapping summary
- missing mappings
- needs-weights mappings
- easy-without-weights mappings
- interpretation reports

This is a diagnostic step before applying weighted HS harmonization.
"""

from __future__ import annotations

from datetime import datetime

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.paths import resolve_log_path
from vacoda.hs_sync.workflow import run_target_hs_mapping_coverage_audit
from vacoda.utils.logger import get_logger, parse_log_level, setup_logging


def main() -> None:
    cfg = HsSyncConfig.from_yaml()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(cfg.logging.level)

    log_path = resolve_log_path(
        cfg.logging.file,
        timestamp=run_id if cfg.logging.timestamp else None,
    )

    setup_logging(level=log_level, log_file=log_path)

    logger = get_logger("SCRIPT.HS_SYNC")
    logger.info("Starting HS-sync target-HS mapping coverage audit.")
    logger.info("Coverage target HS: %s", cfg.coverage.target_hs)
    logger.info(
        "Coverage year range: %s–%s",
        cfg.coverage.min_year,
        cfg.coverage.max_year,
    )
    logger.debug("Log file: %s", log_path)

    try:
        outputs = run_target_hs_mapping_coverage_audit(run_id=run_id)
    except Exception:
        logger.exception("HS-sync target-HS mapping coverage audit failed.")
        logger.info("Log file: %s", log_path)
        raise

    logger.info("Finished HS-sync target-HS mapping coverage audit.")
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