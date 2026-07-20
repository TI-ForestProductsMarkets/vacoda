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
from vacoda.utils.timing import timed_step, timed_workflow


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

def _artifact_result(
    *,
    path_key: str,
    path: Path) -> dict[str, Any]:
    """
    Return a standard result for a disabled step.

    When the expected artifact already exists, downstream steps may use
    it. Otherwise, the step remains skipped and downstream dependency
    checks will fail with a clear message when needed.
    """
    return {
        "status": "provided" if path.exists() else "skipped",
        path_key: path,
        "output_path": path if path.exists() else None,
    }


def _require_artifact(
    *,
    result: dict[str, Any],
    path_key: str,
    consumer_step: str,
    producer_product: str) -> Path:
    """Return a required upstream artifact or raise a clear error."""
    path = result.get(path_key)

    if not isinstance(path, Path):
        raise RuntimeError(
            f"{consumer_step} requires {producer_product}, but the "
            f"upstream result does not contain {path_key!r}."
        )

    if not path.exists():
        raise FileNotFoundError(
            f"{consumer_step} requires {producer_product}, but it was "
            f"not found at {path}. Enable the corresponding product or "
            "provide the existing artifact."
        )

    return path


def _log_step_header(
    *,
    logger,
    symbol: str,
    title: str) -> None:
    logger.info("=" * 72)
    logger.info("%s %s", symbol, title)
    logger.info("=" * 72)


def _resolve_primary_estimation_name(
    *,
    estimated_rates_result: dict[str, Any],
) -> str | None:
    """Resolve full primary estimation run name from Step 5 metadata."""
    value = estimated_rates_result.get("primary_estimation_run")

    if value and "__" in str(value):
        return str(value)

    return None


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

    info = get_symbols("info")
    build = get_symbols("construction")
    ok = get_symbols("check")
    warn = get_symbols("warning")

    logger.info("=" * 72)
    logger.info("%s START — CIF/FOB workflow", info)
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
        "Enabled products: "
        "prepared_input=%s mirror_pairs=%s training_sample=%s "
        "training_features=%s estimated_rates=%s diagnostics=%s corrected_trade=%s",
        cfg.products.prepared_input.enable,
        cfg.products.mirror_pairs.enable,
        cfg.products.training_sample.enable,
        cfg.products.training_features.enable,
        cfg.products.estimated_rates.enable,
        cfg.products.diagnostics.enable,
        cfg.products.corrected_trade.enable,
    )
    logger.info(
        "Decision report: enabled=%s kind=%s subdir=%s "
        "filename=%s timestamp=%s write_latest=%s",
        cfg.artifacts.enable,
        cfg.artifacts.kind,
        cfg.artifacts.subdir,
        cfg.artifacts.filename,
        cfg.artifacts.timestamp,
        cfg.artifacts.write_latest,
    )
    logger.info("=" * 72)

    # ------------------------------------------------------------------
    # Default results also expose expected artifact paths, allowing a
    # disabled step to be supplied by an already existing file.
    # ------------------------------------------------------------------

    prepared_result = _artifact_result(
        path_key="prepared_input_path",
        path=cfg.product_path("prepared_input"),
    )

    mirror_result = _artifact_result(
        path_key="mirror_pairs_path",
        path=cfg.product_path("mirror_pairs"),
    )

    training_sample_result = _artifact_result(
        path_key="training_sample_path",
        path=cfg.product_path("training_sample"),
    )

    training_features_result = _artifact_result(
        path_key="training_features_path",
        path=cfg.product_path("training_features"),
    )

    training_evidence_audit_result = _artifact_result(
        path_key="training_evidence_audit_path",
        path=cfg.product_path("training_evidence_audit"),
    )

    estimated_rates_result = _artifact_result(
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

    with timed_workflow(logger, "CIF/FOB workflow"):
        STEP =1
        # ==================================================================
        # STEP 1 — Prepare canonical CIF/FOB input
        # ==================================================================
        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP-1], total_steps=TOTAL_WORKFLOW_STEPS):
            
            if cfg.products.prepared_input.enable:
                _log_step_header(
                    logger=logger,
                    symbol=build,
                    title=WORKFLOW_STEPS[STEP-1],
                )

                prepared_result = prepare_cif_fob_input(
                    cfg=cfg,
                    force=force,
                    logger=logger,
                )

                if prepared_result.get("audit_passed", False):
                    logger.info(
                        "%s CIF/FOB input preparation audit passed.",
                        ok,
                    )
                else:
                    logger.warning(
                    "%s CIF/FOB input preparation audit did not pass. "
                    "Check the Step 1 logs.",
                    warn,
                )
            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP-1],
                )
            
            STEP += 1

        # ==================================================================
        # STEP 2 — Build bilateral mirror pairs
        # ==================================================================

        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP-1], total_steps=TOTAL_WORKFLOW_STEPS):
            if cfg.products.mirror_pairs.enable:
                _log_step_header(
                    logger=logger,
                    symbol=build,
                    title=WORKFLOW_STEPS[STEP-1],
                )
                prepared_input_path = _require_artifact(
                    result=prepared_result,
                    path_key="prepared_input_path",
                    consumer_step="Mirror-pair step",
                    producer_product="prepared-input Parquet",
                )

                mirror_result = build_mirror_pairs(
                                    cfg=cfg,
                                    input_path=prepared_input_path,
                                    force=force,
                                    logger=logger,
                                )
            else:
                logger.info(
                        "%s SKIP — %s (disabled in config)",
                        warn,
                        WORKFLOW_STEPS[STEP-1],
                    )

            STEP += 1
        
        # ==================================================================
        # STEP 3 — Build factor-candidate training sample
        # ==================================================================

        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP-1], total_steps=TOTAL_WORKFLOW_STEPS):
            if cfg.products.training_sample.enable:
                _log_step_header(
                logger=logger,
                symbol=build,
                title=WORKFLOW_STEPS[STEP-1],
                )

                mirror_pairs_path = _require_artifact(
                        result=mirror_result,
                        path_key="mirror_pairs_path",
                        consumer_step="Training-sample step",
                        producer_product="mirror-pairs Parquet",
                    )

                training_sample_result = build_training_sample(
                    cfg=cfg,
                    input_path=mirror_pairs_path,
                    force=force,
                    logger=logger,
                )
            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP-1],
                )
                
            STEP += 1
        
        # ==================================================================
        # STEP 4 — Enrich training sample with predictors
        # ==================================================================

        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP-1], total_steps=TOTAL_WORKFLOW_STEPS):
            if cfg.products.training_features.enable:
                _log_step_header(
                    logger=logger,
                    symbol=build,
                    title=WORKFLOW_STEPS[STEP-1],
                )

                training_sample_path = _require_artifact(
                    result=training_sample_result,
                    path_key="training_sample_path",
                    consumer_step="Training-features step",
                    producer_product="training-sample Parquet",
                )

                training_features_result = build_training_features(
                    cfg=cfg,
                    input_path=training_sample_path,
                    force=force,
                    logger=logger,
                )
            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP-1],
                )

            STEP += 1

        # ==================================================================
        # STEP 5 — Audit training-evidence representativeness
        # ==================================================================

        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP - 1], total_steps=TOTAL_WORKFLOW_STEPS,):
            if cfg.products.training_evidence_audit.enable:
                _log_step_header(
                    logger=logger,
                    symbol=build,
                    title=WORKFLOW_STEPS[STEP - 1],
                )

                training_features_path = _require_artifact(
                    result=training_features_result,
                    path_key="training_features_path",
                    consumer_step="Training-evidence audit step",
                    producer_product="training-features Parquet",
                )

                training_evidence_audit_result = (
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
                    ok,
                    training_evidence_audit_result[
                        "training_evidence_audit_path"
                    ],
                )
                logger.info(
                    "Training-evidence audit rows: %s",
                    training_evidence_audit_result["n_rows"],
                )

            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP - 1],
                )

            STEP += 1

        
        # ==================================================================
        # STEP 6 — Estimate CIF/FOB correction rates
        # ==================================================================

        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP-1], total_steps=TOTAL_WORKFLOW_STEPS):
            if cfg.products.estimated_rates.enable:
                _log_step_header(
                    logger=logger,
                    symbol=build,
                    title=WORKFLOW_STEPS[STEP-1],
                    )

                training_features_path = _require_artifact(
                        result=training_features_result,
                        path_key="training_features_path",
                        consumer_step="Rate-estimation step",
                        producer_product="training-features Parquet",
                    )
                
                estimated_rates_result = estimate_cif_fob_rates(
                        cfg=cfg,
                        input_path=training_features_path,
                        force=force,
                        logger=logger,
                    )
            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP-1],
                )
            STEP += 1

        # ==================================================================
        # STEP 7 — Build CIF/FOB diagnostics
        # ==================================================================

        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP-1], total_steps=TOTAL_WORKFLOW_STEPS):
            if cfg.products.diagnostics.enable:
                _log_step_header(
                    logger=logger,
                    symbol=build,
                    title=WORKFLOW_STEPS[STEP-1],
                )

                estimated_rates_path = _require_artifact(
                        result=estimated_rates_result,
                        path_key="estimated_rates_path",
                        consumer_step="Diagnostics step",
                        producer_product="estimated-rates Parquet",
                    )

                diagnostics_result = write_primary_holdout_diagnostics(
                                            cfg=cfg,
                                            estimated_rates_path=estimated_rates_path,
                                        )

                logger.info(
                        "%s CIF/FOB diagnostics written: %s",
                        ok,
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

            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP-1],
                )
            STEP += 1

        #=================================================================
        # STEP 8 — Apply CIF/FOB corrections to trade data
        #=================================================================
        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP-1], total_steps=TOTAL_WORKFLOW_STEPS):
            if cfg.products.corrected_trade.enable:
                _log_step_header(
                    logger=logger,
                    symbol=build,
                    title=WORKFLOW_STEPS[STEP-1],
                )

                estimated_rates_path = _require_artifact(
                    result=estimated_rates_result,
                    path_key="estimated_rates_path",
                    consumer_step="Trade-correction step",
                    producer_product="estimated-rates Parquet",
                )

                corrected_trade_result = apply_cif_fob_weights(
                    cfg=cfg,
                    estimated_rates_path=estimated_rates_path,
                    logger=logger,
                )

                # Placeholder for the trade-correction step
                logger.info(
                        "%s Corrected CIF/FOB trade written: %s",
                        ok,
                        corrected_trade_result["output_path"],
                    )
            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP-1],
                )
            STEP += 1

        # ==================================================================
        # STEP 9 — Write human-readable decision report
        # ==================================================================

        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP-1], total_steps=TOTAL_WORKFLOW_STEPS):
            if cfg.artifacts.enable:
                _log_step_header(
                    logger=logger,
                    symbol=build,
                    title=WORKFLOW_STEPS[STEP-1],
                )

                _require_artifact(
                    result=training_features_result,
                    path_key="training_features_path",
                    consumer_step="CIF/FOB reporting step",
                    producer_product="training-features Parquet",
                )

                _require_artifact(
                    result=estimated_rates_result,
                    path_key="estimated_rates_path",
                    consumer_step="CIF/FOB reporting step",
                    producer_product="estimated-rates Parquet",
                )

                if cfg.products.diagnostics.enable:
                    _require_artifact(
                        result=diagnostics_result,
                        path_key="diagnostics_path",
                        consumer_step="CIF/FOB reporting step",
                        producer_product="diagnostics Parquet",
                    )

                report_result = write_cif_fob_decision_report(
                    cfg=cfg,
                    run_id=run_id,
                    training_features_result=training_features_result,
                    estimated_rates_result=estimated_rates_result,
                    diagnostics_result=diagnostics_result,
                    training_evidence_audit_result=training_evidence_audit_result,
                    corrected_trade_result=corrected_trade_result,
                    logger=logger,
                )
            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP-1],
                )
            STEP += 1

    # ------------------------------------------------------------------
    # The latest output is an orchestration detail only. Readiness and
    # audit decisions remain inside the individual step results.
    # ------------------------------------------------------------------

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

    latest_output_path: Path | None = None

    for result, path_key in [
        (diagnostics_result, "diagnostics_path"),
        (corrected_trade_result, "corrected_trade_path"),
        (estimated_rates_result, "estimated_rates_path"),
        (training_features_result, "training_features_path"),
        (training_sample_result, "training_sample_path"),
        (mirror_result, "mirror_pairs_path"),
        (prepared_result, "prepared_input_path"),
    ]:
        candidate = result.get(path_key)

        if isinstance(candidate, Path) and candidate.exists():
            latest_output_path = candidate
            break

    logger.info("=" * 72)
    logger.info("%s CIF/FOB workflow finished.", info)
    logger.info("Latest data output: %s", latest_output_path)
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
