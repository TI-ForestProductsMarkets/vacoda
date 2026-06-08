"""
Script to audit HS system coverage by generating a report on how many HS records are present in the source data and 
how many are successfully processed through the workflow. This helps identify gaps in coverage and areas for improvement 
in the HS Sync pipeline.
"""
from datetime import datetime
from vacoda.config.hs_coverage import (
    get_logging_file,
    get_logging_level,
    get_logging_timestamp_bool,
    load_hs_coverage_config,
)
from vacoda.config.paths import resolve_log_path
from vacoda.hs_coverage.workflow import run_hs_system_coverage_audit
from vacoda.utils.logger import get_logger, parse_log_level, setup_logging


def main() -> None:
    cfg = load_hs_coverage_config()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(get_logging_level(cfg))
    log_path = resolve_log_path(
        get_logging_file(cfg),
        timestamp=run_id,
    )

    setup_logging(level=log_level, log_file=log_path)

    logger = get_logger("SCRIPT.HS_COVERAGE")
    logger.info("Starting HS Coverage audit.")
    logger.debug("Log file: %s", log_path)

    try:
        outputs = run_hs_system_coverage_audit(run_id=run_id)
    except Exception as e:
        logger.exception("HS Coverage audit failed.", exc_info=e)
        logger.info("Log file: %s", log_path)
        raise

    logger.info("Finished HS Coverage audit.")
    logger.info("\nHS Coverage Workflow complete.")
    logger.info(f"  Output directory: {outputs.get('output_dir')}")
    logger.info(f"  Report: {outputs.get('report')}")
    logger.info(f"  Log file: {log_path}")

if __name__ == "__main__":
    main()