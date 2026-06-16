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

WORKFLOW_STEPS = ["STEP 1 — Prepare CIF/FOB input",
                  "STEP 2 — Build CIF/FOB mirror pairs",
                  "STEP 3 — Build CIF/FOB training sample"]

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
    logger.info(
        "Enabled products: prepared_input=%s mirror_pairs=%s training_sample=%s estimated_rates=%s corrected_trade=%s",
        cfg.products.prepared_input.enable,
        cfg.products.mirror_pairs.enable,
        cfg.products.training_sample.enable,
        cfg.products.estimated_rates.enable,
        cfg.products.corrected_trade.enable,
    )
    logger.info("=" * 72)
    prepared_path = cfg.product_path("prepared_input")
    mirror_path = cfg.product_path("mirror_pairs")
    training_path = cfg.product_path("training_sample")

    prepared_result: dict[str, Any] = {
        "status": "skipped",
        "prepared_input_path": prepared_path,
    }
    mirror_result: dict[str, Any] = {
        "status": "skipped",
        "mirror_pairs_path": mirror_path,
        "n_mirror_groups": None,
        "n_with_both_mirrors": None,
    }
    training_result: dict[str, Any] = {
        "status": "skipped",
        "training_sample_path": training_path,
        "n_high_weight_concordance": None,
        "n_direct_factors": None,
        "n_mirror_factors": None,
        "n_model_eligible": None,
        "median_observed_factor": None,
        "median_observed_factor_high_weight_concordance": None,
    }

    #### STEP 1: Prepare CIF/FOB input ###########################################

    if cfg.products.prepared_input.enable:
        logger.info("=" * 72)
        logger.info(WORKFLOW_STEPS[0])
        logger.info("=" * 72)
        prepared_result = prepare_cif_fob_input(
            cfg=cfg,
            force=force,
            logger=logger,
        )
    else:
        logger.info("SKIP — %s (disabled in config)", WORKFLOW_STEPS[0])


    #### STEP 2: Build CIF/FOB mirror pairs ###################################
    if cfg.products.mirror_pairs.enable:
        logger.info("=" * 72)
        logger.info(WORKFLOW_STEPS[1])
        logger.info("=" * 72)
        if not prepared_result["prepared_input_path"].exists():
            raise FileNotFoundError(
                "Mirror-pair step requires prepared input parquet, but it was not found at "
                f"{prepared_result['prepared_input_path']}. "
                "Enable cif_fob.products.prepared_input or provide the prepared input artifact."
            )

        mirror_result = build_mirror_pairs(
            cfg=cfg,
            input_path=prepared_result["prepared_input_path"],
            force=force,
            logger=logger,
        )
    else:
        logger.info("SKIP — %s (disabled in config)", WORKFLOW_STEPS[1])


    #### STEP 3: Build CIF/FOB training sample #############################
    if cfg.products.training_sample.enable:
        logger.info("=" * 72)
        logger.info(WORKFLOW_STEPS[2])
        logger.info("=" * 72)
        if not mirror_result["mirror_pairs_path"].exists():
            raise FileNotFoundError(
                "Training-sample step requires mirror-pairs parquet, but it was not found at "
                f"{mirror_result['mirror_pairs_path']}. "
                "Enable cif_fob.products.mirror_pairs or provide the mirror-pairs artifact."
            )

        training_result = build_training_sample(
            cfg=cfg,
            input_path=mirror_result["mirror_pairs_path"],
            force=force,
            logger=logger,
        )
    else:
        logger.info("SKIP — %s (disabled in config)", WORKFLOW_STEPS[2])

    if cfg.products.estimated_rates.enable:
        logger.warning(
            "ENABLE requested for estimated_rates, but this workflow does not implement rate estimation yet."
        )

    if cfg.products.corrected_trade.enable:
        logger.warning(
            "ENABLE requested for corrected_trade, but this workflow does not implement trade correction yet."
        )

    ready_for_rate_estimation = (
        cfg.products.mirror_pairs.enable
        and cfg.products.training_sample.enable
        and mirror_result["status"] in {"completed", "reused"}
        and int(training_result.get("n_model_eligible") or 0) > 0
    )

    output_path = training_result["training_sample_path"]
    if not cfg.products.training_sample.enable and cfg.products.mirror_pairs.enable:
        output_path = mirror_result["mirror_pairs_path"]
    if not cfg.products.training_sample.enable and not cfg.products.mirror_pairs.enable:
        output_path = prepared_result["prepared_input_path"]

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
        "output_path": output_path,
        "prepared_input_path": prepared_result["prepared_input_path"],
        "mirror_pairs_path": mirror_result["mirror_pairs_path"],
        "training_sample_path": training_result["training_sample_path"],
        "prepared_input_enabled": cfg.products.prepared_input.enable,
        "mirror_pairs_enabled": cfg.products.mirror_pairs.enable,
        "training_sample_enabled": cfg.products.training_sample.enable,
        "estimated_rates_enabled": cfg.products.estimated_rates.enable,
        "corrected_trade_enabled": cfg.products.corrected_trade.enable,
        "n_mirror_groups": mirror_result.get("n_mirror_groups"),
        "n_with_both_mirrors": mirror_result.get("n_with_both_mirrors"),
        "n_high_weight_concordance": training_result.get("n_high_weight_concordance"),
        "n_direct_factors": training_result.get("n_direct_factors"),
        "n_mirror_factors": training_result.get("n_mirror_factors"),
        "n_model_eligible": training_result.get("n_model_eligible"),
        "median_observed_factor": training_result.get("median_observed_factor"),
        "median_observed_factor_high_weight_concordance": training_result.get(
            "median_observed_factor_high_weight_concordance"
        ),
    }