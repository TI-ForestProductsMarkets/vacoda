from vacoda.config.hs_sync import (
    get_logging_file,
    get_logging_level,
    get_logging_timestamp_bool,
    load_hs_sync_config,
)
from vacoda.config.paths import resolve_log_path
from vacoda.hs_sync.workflow import run_hs_system_coverage_audit
from vacoda.utils.logger import get_logger, parse_log_level, setup_logging


def main() -> None:
    cfg = load_hs_sync_config()

    log_level = parse_log_level(get_logging_level(cfg))
    log_path = resolve_log_path(
        get_logging_file(cfg),
        timestamp=get_logging_timestamp_bool(cfg),
    )

    setup_logging(level=log_level, log_file=log_path)

    logger = get_logger("SCRIPT.HS_SYNC_COVERAGE")
    logger.info("Starting HS system coverage audit.")
    logger.debug("Log file: %s", log_path)

    outputs = run_hs_system_coverage_audit()

    logger.info("Finished HS system coverage audit.")
    logger.info("\nHS System Coverage Workflow complete.")
    logger.info(f"  Output directory: {outputs.get('output_dir')}")
    logger.info(f"  Report: {outputs.get('report')}")
    logger.info(f"  Log file: {log_path}")

if __name__ == "__main__":
    main()