#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime
from pprint import pformat

from vacoda.config.cif_fob import CifFobConfig
from vacoda.config.pipeline import PipelineConfig
from vacoda.cif_fob.workflow import run_cif_fob_workflow
from vacoda.utils.logger import get_logger, setup_logging
import argparse as ap

def main() -> None:
    pipeline_cfg = PipelineConfig.from_yaml()

    cfg = CifFobConfig.from_yaml(
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    setup_logging(level="INFO")

    aprse = ap.ArgumentParser(description="Run CIF/FOB mirror pairs workflow")
    aprse.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of all steps, even if outputs already exist",
    )
    args = aprse.parse_args()

    result = run_cif_fob_workflow(
        cfg=cfg,
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        force=args.force,
        logger=get_logger("CIF_FOB"),
    )

    print(pformat(result, sort_dicts=False))


if __name__ == "__main__":
    main()