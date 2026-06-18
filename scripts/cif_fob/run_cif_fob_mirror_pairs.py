#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime
from pprint import pformat
import argparse as ap

from vacoda.config.cif_fob import CifFobConfig
from vacoda.config.pipeline import PipelineConfig
from vacoda.config.paths import resolve_log_path
from vacoda.cif_fob.workflow import run_cif_fob_workflow
from vacoda.utils.logger import get_logger, parse_log_level, setup_logging

def main() -> None:
    pipeline_cfg = PipelineConfig.from_yaml()

    cfg = CifFobConfig.from_yaml(
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
    logger = get_logger("WORKFLOW.CIF_FOB")

    aprse = ap.ArgumentParser(description="Run CIF/FOB mirror pairs workflow")
    aprse.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of all steps, even if outputs already exist",
    )
    args = aprse.parse_args()

    try:
        result = run_cif_fob_workflow(
            cfg=cfg,
            run_id=run_id,
            force=args.force,
            logger=logger,
        )
    except Exception:
        logger.exception("CIF/FOB workflow failed.")
        logger.info("Log file: %s", log_path)
        raise

    logger.info("CIF/FOB workflow finished.")
    logger.info("  Status: %s", result.get("status"))
    logger.info("  Output path: %s", result.get("output_path"))
    logger.info("  Log file: %s", log_path)

    print(pformat(result, sort_dicts=False))


if __name__ == "__main__":
    main()