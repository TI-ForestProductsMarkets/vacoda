"""
vacoda.cif_fob.workflow

Workflow orchestration for the CIF/FOB correction stage.

Each processing function owns its own audits, readiness checks, and
step-specific metrics. This module only:

1. resolves step dependencies;
2. runs enabled steps in order;
3. passes each produced artifact to the next step;
4. returns the individual step results without re-deriving them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vacoda.cif_fob.estimate_rates import estimate_cif_fob_rates
from vacoda.cif_fob.mirror_pairs import build_mirror_pairs
from vacoda.cif_fob.prepare_input import prepare_cif_fob_input
from vacoda.cif_fob.reporting import write_cif_fob_decision_report
from vacoda.cif_fob.training_features import build_training_features
from vacoda.cif_fob.training_evidence_audit import write_training_evidence_audit
from vacoda.cif_fob.training_sample import build_training_sample
from vacoda.cif_fob.diagnostics import write_primary_holdout_diagnostics
from vacoda.cif_fob.apply_rates import apply_cif_fob_weights
from vacoda.config.cif_fob import CifFobConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.logger import get_logger
from vacoda.utils.runtime import run_step, timed_workflow
from vacoda.utils.artifacts import artifact_result, require_artifact, require_enabled_products, require_product_dependency

WORKFLOW_STEPS = [
    "Prepare CIF/FOB input",
    "Build CIF/FOB mirror pairs",
    "Build CIF/FOB training sample",
    "Build CIF/FOB training features",
    "Audit CIF/FOB training evidence",
    "Estimate CIF/FOB correction rates",
    "Build CIF/FOB diagnostics",
    "Apply CIF/FOB corrections to trade data",
    "Write CIF/FOB decision report",
]

TOTAL_WORKFLOW_STEPS = len(WORKFLOW_STEPS)

REQUIRED_PRODUCTS = (
    "prepared_input",
    "mirror_pairs",
    "training_sample",
    "training_features",
    "training_evidence_audit",
    "estimated_rates",
    "diagnostics",
    "corrected_trade",
)


def get_enabled_products(cfg: CifFobConfig) -> dict[str, bool]:
    """Return CIF/FOB product enablement as a plain dictionary."""
    return {
        "prepared_input": cfg.products.prepared_input.enable,
        "mirror_pairs": cfg.products.mirror_pairs.enable,
        "training_sample": cfg.products.training_sample.enable,
        "training_features": cfg.products.training_features.enable,
        "training_evidence_audit": (
            cfg.products.training_evidence_audit.enable
        ),
        "estimated_rates": cfg.products.estimated_rates.enable,
        "diagnostics": cfg.products.diagnostics.enable,
        "corrected_trade": cfg.products.corrected_trade.enable,
        "decision_report": cfg.artifacts.enable,
    }


def run_cif_fob_workflow(
    *,
    cfg: CifFobConfig,
    run_id: str,
    force: bool = False,
    logger=None,
) -> dict[str, Any]:
    """
    Run the enabled CIF/FOB workflow steps in dependency order.

    Step-specific functions remain responsible for their own:
      - audits;
      - readiness decisions;
      - metrics;
      - output validation.

    The workflow does not recalculate or reinterpret those results.
    """
    logger = logger or get_logger("CIF_FOB")

    # ------------------------------------------------------------------
    # Default results also expose expected artifact paths, allowing a
    # disabled step to be supplied by an already existing file.
    # ------------------------------------------------------------------

    prepared_result = artifact_result(
            path_key="prepared_input_path",
            path=cfg.product_path("prepared_input"),
        )

    mirror_result = artifact_result(
            path_key="mirror_pairs_path",
            path=cfg.product_path("mirror_pairs"),
        )

    training_sample_result = artifact_result(
            path_key="training_sample_path",
            path=cfg.product_path("training_sample"),
        )

    training_features_result = artifact_result(
            path_key="training_features_path",
            path=cfg.product_path("training_features"),
        )

    training_evidence_audit_result = artifact_result(
            path_key="training_evidence_audit_path",
            path=cfg.product_path("training_evidence_audit"),
        )

    estimated_rates_result = artifact_result(
            path_key="estimated_rates_path",
            path=cfg.product_path("estimated_rates"),
        )
        
    diagnostics_result: dict[str, Any] = {
            "status": "skipped",
            "diagnostics_path": None,
            "output_path": None,
        }

    report_result: dict[str, Any] = {
            "status": "disabled" if not cfg.artifacts.enable else "skipped",
            "report_path": None,
            "latest_report_path": None,
        }

    corrected_trade_result: dict[str, Any] = {
                "status": (
                    "skipped"
                    if cfg.products.corrected_trade.enable
                    else "disabled"
                ),
                "output_path": None,
                "audit_path": None,
                "corrected_trade_path": None,
                "corrected_trade_audit_path": None,
                "ready_for_trade_sync": False,
            }

    enabled_products = get_enabled_products(cfg)

    require_enabled_products(
        workflow_name="CIF/FOB", 
        enabled_products=enabled_products,
        required_products=REQUIRED_PRODUCTS,
    )

    require_product_dependency(
        workflow_name="CIF/FOB",
        enabled_products=enabled_products,
        product="decision_report",
        required_product="diagnostics",
    )


    with timed_workflow(logger, "CIF/FOB"):

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
            "Enabled products: %s",
            " ".join(
                f"{name}={enabled}"
                for name, enabled in enabled_products.items()
            ),
        )

        # ==================================================================
        # STEP 1 — Prepare canonical CIF/FOB input
        # ==================================================================
            
        prepared_result = run_step(
            logger=logger,
            step_number=1,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[0],
            runner = lambda: prepare_cif_fob_input(
                    cfg=cfg,
                    force=force,
                    logger=logger,
                ),
        )

        if prepared_result.get("audit_passed", False):
            logger.info(
                        "%s CIF/FOB input preparation audit passed.",
                        get_symbols("check"),
                    )
        else:
            logger.warning(
                    "%s CIF/FOB input preparation audit did not pass. "
                    "Check the Step 1 logs.",
                    get_symbols("warn"),
                )

        # ==================================================================
        # STEP 2 — Build bilateral mirror pairs
        # ==================================================================

        prepared_input_path = require_artifact(
                    result=prepared_result,
                    path_key="prepared_input_path",
                    consumer_step="Mirror-pair step",
                    producer_product="prepared-input Parquet",
                )

        mirror_result = run_step(
            logger=logger,
            step_number=2,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[1],
            runner=lambda: build_mirror_pairs(
                                    cfg=cfg,
                                    input_path=prepared_input_path,
                                    force=force,
                                    logger=logger,
                                )
        )

        
        # ==================================================================
        # STEP 3 — Build factor-candidate training sample
        # ==================================================================

        mirror_pairs_path = require_artifact(
                        result=mirror_result,
                        path_key="mirror_pairs_path",
                        consumer_step="Training-sample step",
                        producer_product="mirror-pairs Parquet",
                    )

        training_sample_result = run_step(
                    logger=logger,
                    step_number=3,
                    total_steps=TOTAL_WORKFLOW_STEPS,
                    title=WORKFLOW_STEPS[2],
                    runner=lambda: build_training_sample(
                                cfg=cfg,
                                input_path=mirror_pairs_path,
                                force=force,
                                logger=logger,
                            )
                )
        
        # ==================================================================
        # STEP 4 — Enrich training sample with predictors
        # ==================================================================

        training_sample_path = require_artifact(
                    result=training_sample_result,
                    path_key="training_sample_path",
                    consumer_step="Training-features step",
                    producer_product="training-sample Parquet",
                )

        training_features_result = run_step(
                    logger=logger,
                    step_number=4,
                    total_steps=TOTAL_WORKFLOW_STEPS,
                    title=WORKFLOW_STEPS[3],
                    runner=lambda: build_training_features(
                                cfg=cfg,
                                input_path=training_sample_path,
                                force=force,
                                logger=logger,
                            )
                )

        # ==================================================================
        # STEP 5 — Audit training-evidence representativeness
        # ==================================================================

        training_features_path = require_artifact(
                    result=training_features_result,
                    path_key="training_features_path",
                    consumer_step="Training-evidence audit step",
                    producer_product="training-features Parquet",
                )

        training_evidence_audit_result = run_step(
                    logger=logger,
                    step_number=5,
                    total_steps=TOTAL_WORKFLOW_STEPS,
                    title=WORKFLOW_STEPS[4],
                    runner=lambda:
                    write_training_evidence_audit(
                        input_path=training_features_path,
                        output_path=cfg.product_path(
                            "training_evidence_audit"
                        ),
                        threads=cfg.engine.threads,
                        logger=logger
                    )
                )

        logger.info(
                    "%s CIF/FOB training-evidence audit written: %s",
                    get_symbols("check"),
                    training_evidence_audit_result[
                        "training_evidence_audit_path"
                    ],
                )
        logger.info(
                    "Training-evidence audit rows: %s",
                    training_evidence_audit_result["n_rows"],
                )

        
        # ==================================================================
        # STEP 6 — Estimate CIF/FOB correction rates
        # ==================================================================

        training_features_path = require_artifact(
                        result=training_features_result,
                        path_key="training_features_path",
                        consumer_step="Rate-estimation step",
                        producer_product="training-features Parquet",
                    )
                
        estimated_rates_result = run_step(
                    logger=logger,
                    step_number=6,
                    total_steps=TOTAL_WORKFLOW_STEPS,
                    title=WORKFLOW_STEPS[5],
                    runner=lambda:estimate_cif_fob_rates(
                            cfg=cfg,
                            input_path=training_features_path,
                            force=force,
                            logger=logger,
                        )
        )

        # ==================================================================
        # STEP 7 — Build CIF/FOB diagnostics
        # ==================================================================

        estimated_rates_path = require_artifact(
                        result=estimated_rates_result,
                        path_key="estimated_rates_path",
                        consumer_step="Diagnostics step",
                        producer_product="estimated-rates Parquet",
                    )

        diagnostics_result = run_step(
                    logger=logger,
                    step_number=7,
                    total_steps=TOTAL_WORKFLOW_STEPS,
                    title=WORKFLOW_STEPS[6],
                    runner=lambda: write_primary_holdout_diagnostics(
                                            cfg=cfg,
                                            estimated_rates_path=estimated_rates_path,
                                            force=force,
                                            logger=logger,
                                        )
        )

        logger.info(
                        "%s CIF/FOB diagnostics written: %s",
                        get_symbols("check"),
                        diagnostics_result["diagnostics_path"],
                    )

        logger.info(
                        "Diagnostics rows: %s; years: %s-%s; "
                        "median APE: %.4f; mean APE: %.4f; P90 APE: %.4f",
                        diagnostics_result["n_rows"],
                        diagnostics_result["year_min"],
                        diagnostics_result["year_max"],
                        diagnostics_result["median_ape"],
                        diagnostics_result["mean_ape"],
                        diagnostics_result["p90_ape"],
                    )                                            


        #=================================================================
        # STEP 8 — Apply CIF/FOB corrections to trade data
        #=================================================================
        estimated_rates_path = require_artifact(
                    result=estimated_rates_result,
                    path_key="estimated_rates_path",
                    consumer_step="Trade-correction step",
                    producer_product="estimated-rates Parquet",
                )

        corrected_trade_result = run_step(
            logger=logger,
            step_number=8,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[7],
            runner=lambda: apply_cif_fob_weights(
                    cfg=cfg,
                    estimated_rates_path=estimated_rates_path,
                    force=force,
                    logger=logger,
                )
        )

        # Placeholder for the trade-correction step
        logger.info(
                        "%s Corrected CIF/FOB trade written: %s",
                        get_symbols("check"),
                        corrected_trade_result["output_path"],
                    )


        # ==================================================================
        # STEP 9 — Write human-readable decision report
        # ==================================================================

        require_artifact(
                    result=training_features_result,
                    path_key="training_features_path",
                    consumer_step="CIF/FOB reporting step",
                    producer_product="training-features Parquet",
                )

        require_artifact(
                    result=estimated_rates_result,
                    path_key="estimated_rates_path",
                    consumer_step="CIF/FOB reporting step",
                    producer_product="estimated-rates Parquet",
                )

        report_result = run_step(
            logger=logger,
            step_number=9,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[8],
            runner=lambda: write_cif_fob_decision_report(
                    cfg=cfg,
                    run_id=run_id,
                    training_features_result=training_features_result,
                    estimated_rates_result=estimated_rates_result,
                    diagnostics_result=diagnostics_result,
                    training_evidence_audit_result=training_evidence_audit_result,
                    corrected_trade_result=corrected_trade_result,
                    logger=logger,
                )
            )


    final_parquet = corrected_trade_result.get(
        "corrected_trade_path"
    ) or corrected_trade_result.get(
        "output_path"
    )

    final_audit = corrected_trade_result.get(
        "corrected_trade_audit_path"
    ) or corrected_trade_result.get(
        "audit_path"
    )

    if final_parquet is not None:
        final_parquet = Path(
            final_parquet
        )

    if final_audit is not None:
        final_audit = Path(
            final_audit
        )

    ready_for_trade_sync = bool(
            corrected_trade_result.get(
                "ready_for_trade_sync",
                False,
            )
        )
    
    if ready_for_trade_sync:
        if final_parquet is None or not final_parquet.exists():
            raise RuntimeError(
                "CIF/FOB correction reported ready_for_trade_sync=True, "
                "but the corrected-trade Parquet does not exist."
            )

        if final_audit is None or not final_audit.exists():
            raise RuntimeError(
                "CIF/FOB correction reported ready_for_trade_sync=True, "
                "but the corrected-trade audit does not exist."
            )

    logger.info("=" * 72)
    logger.info("%s CIF/FOB workflow finished.", get_symbols("info"))
    logger.info(
        "Final corrected-trade output: %s",
        final_parquet,
    )
    logger.info(
        "Final corrected-trade audit: %s",
        final_audit,
    )
    logger.info(
        "Ready for Trade Sync: %s",
        ready_for_trade_sync,
    )
    logger.info(
        "Decision report: %s",
        report_result.get("report_path"),
    )
    logger.info("=" * 72)

    return {
        "status": "completed",
        "run_id": run_id,
        "ready_for_next_step": ready_for_trade_sync,
        "ready_for_trade_sync": ready_for_trade_sync,

        "output_path": final_parquet,
        "audit_path": final_audit,
        "corrected_trade_path": final_parquet,
        "corrected_trade_audit_path": final_audit,
        
        "n_failed_audit_rows": corrected_trade_result.get(
            "n_failed_audit_rows"
        ),
        "prepared_input_path": prepared_result.get(
            "prepared_input_path"
        ),
        "mirror_pairs_path": mirror_result.get(
            "mirror_pairs_path"
        ),
        "training_sample_path": training_sample_result.get(
            "training_sample_path"
        ),
        "training_features_path": training_features_result.get(
            "training_features_path"
        ),
        "training_evidence_audit_path": training_evidence_audit_result.get(
            "training_evidence_audit_path"
        ),
        "estimated_rates_path": estimated_rates_result.get(
            "estimated_rates_path"
        ),
        "diagnostics_path": diagnostics_result.get(
            "diagnostics_path"
        ),
        "decision_report_path": report_result.get(
            "report_path"
        ),
        "latest_decision_report_path": report_result.get(
            "latest_report_path"
        ),

        "products_enabled": {
            "prepared_input": cfg.products.prepared_input.enable,
            "mirror_pairs": cfg.products.mirror_pairs.enable,
            "training_sample": cfg.products.training_sample.enable,
            "training_features": cfg.products.training_features.enable,
            "training_evidence_audit": cfg.products.training_evidence_audit.enable,
            "estimated_rates": cfg.products.estimated_rates.enable,
            "corrected_trade": cfg.products.corrected_trade.enable,
            "diagnostics": cfg.products.diagnostics.enable,
            "decision_report": cfg.artifacts.enable,
        },
        "steps": {
            "prepared_input": prepared_result,
            "mirror_pairs": mirror_result,
            "training_sample": training_sample_result,
            "training_features": training_features_result,
            "training_evidence_audit": training_evidence_audit_result,
            "estimated_rates": estimated_rates_result,
            "diagnostics": diagnostics_result,
            "corrected_trade": corrected_trade_result,
            "decision_report": report_result,
        },
    }
