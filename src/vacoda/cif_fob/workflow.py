"""
vacoda.cif_fob.workflow

Workflow orchestration for the CIF/FOB correction stage.
"""

from __future__ import annotations

from typing import Any

from vacoda.cif_fob.mirror_pairs import build_mirror_pairs
from vacoda.cif_fob.prepare_input import prepare_cif_fob_input
from vacoda.config.cif_fob import CifFobConfig
from vacoda.utils import logger
from vacoda.utils.logger import get_logger


def run_cif_fob_workflow(
    *,
    cfg: CifFobConfig,
    run_id: str,
    logger=None,
) -> dict[str, Any]:
    """Run the CIF/FOB workflow.

    Initial implementation:
      1. Build canonical import/export mirror pairs.

    Model estimation and trade correction are added in later stages.
    """
    logger = logger or get_logger("CIF_FOB")

    logger.info("=" * 72)
    logger.info("START — CIF/FOB workflow")
    logger.info("=" * 72)
    logger.info("Run ID: %s", run_id)
    logger.info("Target HS: %s", cfg.scope.target_hs)
    logger.info(
        "Year range: %s-%s",
        cfg.scope.min_year,
        cfg.scope.max_year,
    )
    logger.info("Input: %s", cfg.input_path())
    logger.info("Output directory: %s", cfg.output_dir())
    logger.info("=" * 72)
    logger.info("STEP 1 — Prepare CIF/FOB input")
    logger.info("=" * 72)

    prepared_result = prepare_cif_fob_input(
        cfg=cfg,
        logger=logger,
    )

    logger.info("=" * 72)
    logger.info("STEP 2 — Build CIF/FOB mirror pairs")
    logger.info("=" * 72)

    mirror_result = build_mirror_pairs(
        cfg=cfg,
        input_path=prepared_result["prepared_input_path"],
        logger=logger,
    )

    ready_for_rate_estimation = (
        mirror_result["status"] == "completed"
        and mirror_result["n_training_eligible"] > 0
    )

    logger.info("=" * 72)
    logger.info("CIF/FOB mirror-pair workflow finished.")
    logger.info(
        "Ready for rate estimation: %s",
        ready_for_rate_estimation,
    )
    logger.info("=" * 72)

    return {
        "status": "completed",
        "run_id": run_id,
        "ready_for_rate_estimation": ready_for_rate_estimation,
        "output_path": mirror_result["mirror_pairs_path"],
        "mirror_pairs_path": mirror_result["mirror_pairs_path"],
        "n_mirror_groups": mirror_result["n_mirror_groups"],
        "n_with_both_mirrors": mirror_result["n_with_both_mirrors"],
        "n_training_eligible": mirror_result["n_training_eligible"],
        "median_raw_value_ratio": mirror_result[
            "median_raw_value_ratio"
        ],
        "median_unit_value_ratio": mirror_result[
            "median_unit_value_ratio"
        ],
        "median_weight_concordance": mirror_result[
            "median_weight_concordance"
        ],
    }