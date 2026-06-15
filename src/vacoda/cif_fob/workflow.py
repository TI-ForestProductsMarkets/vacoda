"""
vacoda.cif_fob.workflow

Workflow orchestration for the CIF/FOB correction stage.
"""

from __future__ import annotations

from typing import Any

from vacoda.cif_fob.mirror_pairs import build_mirror_pairs
from vacoda.cif_fob.prepare_input import prepare_cif_fob_input
from vacoda.config.cif_fob import CifFobConfig
from vacoda.cif_fob.training_sample import build_training_sample
from vacoda.utils import logger
from vacoda.utils.logger import get_logger


def run_cif_fob_workflow(
    *,
    cfg: CifFobConfig,
    run_id: str,
    force: bool = False,
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
        force=force,
        logger=logger,
    )

    logger.info("=" * 72)
    logger.info("STEP 2 — Build CIF/FOB mirror pairs")
    logger.info("=" * 72)

    mirror_result = build_mirror_pairs(
        cfg=cfg,
        input_path=prepared_result["prepared_input_path"],
        force=force,
        logger=logger,
    )


    logger.info("=" * 72)
    logger.info("STEP 3 — Build CIF/FOB training sample")
    logger.info("=" * 72)
    training_result = build_training_sample(
            cfg=cfg,
            input_path=mirror_result["mirror_pairs_path"],
            force=force,
            logger=logger,
        )

    ready_for_rate_estimation = (
        mirror_result["status"] == "completed"
        and training_result["n_model_eligible"] > 0
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

    "output_path": training_result["training_sample_path"],

    "prepared_input_path": prepared_result[
        "prepared_input_path"
    ],
    "mirror_pairs_path": mirror_result[
        "mirror_pairs_path"
    ],
    "training_sample_path": training_result[
        "training_sample_path"
    ],

    "n_mirror_groups": mirror_result[
        "n_mirror_groups"
    ],
    "n_with_both_mirrors": mirror_result[
        "n_with_both_mirrors"
    ],
    "n_high_weight_concordance": training_result[
        "n_high_weight_concordance"
    ],
    "n_direct_factors": training_result[
        "n_direct_factors"
    ],
    "n_mirror_factors": training_result[
        "n_mirror_factors"
    ],
    "n_model_eligible": training_result[
        "n_model_eligible"
    ],
    "median_observed_factor": training_result[
        "median_observed_factor"
    ],
}