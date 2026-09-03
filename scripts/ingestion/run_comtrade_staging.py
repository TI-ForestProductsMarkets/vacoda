"""
scripts.dataset.run_comtrade_staging

Script to run the Comtrade staging dataset build process, which reads raw Comtrade data files, 
applies schema, and writes out staged parquet files along with an audit report. 
This script is intended to be run as a standalone process, and it uses the configuration defined in the dataset config YAML file.
"""


from datetime import datetime
from time import perf_counter

from vacoda.config.ingestion import DatasetConfig
from vacoda.config.pipeline import PipelineConfig
from vacoda.steps.ingestion.staging import build_comtrade_ingestion_layer
from vacoda.config.paths import resolve_log_path
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)

def main() -> None:
    pipeline_cfg = PipelineConfig.from_yaml()

    cfg = DatasetConfig.from_yaml(
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

    logger = get_logger("STAGING.COMTRADE")

    step_start = perf_counter()

    try:
        outputs = build_comtrade_ingestion_layer(
                            cfg=cfg, 
                            run_id=run_id,
                            logger=logger)
    except Exception:
        logger.exception("Comtrade staging dataset build failed.")
        logger.info("Log file: %s", log_path)
        raise

    step_end = perf_counter()

    logger.info("Comtrade staging dataset build finished.")
    logger.info("  Duration: %.2f seconds", step_end - step_start)
    logger.info("  Status: %s", outputs.get("status"))
    logger.info("  Ready for HS-sync: %s", outputs.get("ready_for_hs_sync"))
    logger.info("  Staged parquet: %s", outputs.get("staged_parquet"))
    logger.info("  Audit file: %s", outputs.get("audit_file"))
    logger.info("  Log file: %s", log_path)


if __name__ == "__main__":
    main()