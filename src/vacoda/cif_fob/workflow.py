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
from vacoda.cif_fob.training_sample import build_training_sample
from vacoda.cif_fob.diagnostics import write_primary_holdout_diagnostics
from vacoda.config.cif_fob import CifFobConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.logger import get_logger
from vacoda.utils.timing import timed_step, timed_workflow


WORKFLOW_STEPS = [
    "Prepare CIF/FOB input",
    "Build CIF/FOB mirror pairs",
    "Build CIF/FOB training sample",
    "Build CIF/FOB training features",
    "Estimate CIF/FOB correction rates",
    "Build CIF/FOB diagnostics",
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

    estimated_rates_result = _artifact_result(
        path_key="estimated_rates_path",
        path=cfg.product_path("estimated_rates"),
    )
    
    diagnostics_result = _artifact_result(
        path_key="diagnostics_path",
        path=cfg.product_path("diagnostics"),
    )

    report_result: dict[str, Any] = {
        "status": "disabled" if not cfg.artifacts.enable else "skipped",
        "report_path": None,
        "latest_report_path": None,
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
        # STEP 5 — Estimate CIF/FOB correction rates
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
        # STEP 6 — Build CIF/FOB diagnostics
        # ==================================================================

        with timed_step(logger, STEP, WORKFLOW_STEPS[STEP-1], total_steps=TOTAL_WORKFLOW_STEPS):
            if cfg.products.diagnostics.enable:
                _log_step_header(
                    logger=logger,
                    symbol=build,
                    title=WORKFLOW_STEPS[STEP-1],
                )

                estimated_rates_path  = _require_artifact(
                    result=estimated_rates_result,
                    path_key="estimated_rates_path",
                    consumer_step="Diagnostics step",
                    producer_product="estimated-rates Parquet",
                )

                primary_run_name =estimated_rates_result.get("primary_estimation_run")

                if not primary_run_name:
                    primary_run_name = estimated_rates_result.get("primary_model_name")

                if not primary_run_name:
                    raise RuntimeError(
                        "Diagnostics step requires the primary estimation "
                        "run name, but estimated_rates_result does not "
                        "contain 'primary_estimation_run' or "
                        "'primary_model_name'."
                    )

                diagnostics_result = write_primary_holdout_diagnostics(
                        estimated_rates_path=estimated_rates_path,
                        output_path=cfg.product_path("diagnostics"),
                        primary_run_name=str(primary_run_name),
                    )

            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP-1],
                )
            STEP += 1

        # ==================================================================
        # STEP 7 — Write human-readable decision report
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
                    logger=logger,
                )
            else:
                logger.info(
                    "%s SKIP — %s (disabled in config)",
                    warn,
                    WORKFLOW_STEPS[STEP-1],
                )
            STEP += 1


    # Trade correction is not wired into this workflow yet.
    if cfg.products.corrected_trade.enable:
        logger.warning(
            "%s corrected_trade is enabled, but the trade-correction "
            "step has not yet been connected to this workflow.",
            warn,
        )

    # ------------------------------------------------------------------
    # The latest output is an orchestration detail only. Readiness and
    # audit decisions remain inside the individual step results.
    # ------------------------------------------------------------------

    latest_output_path: Path | None = None

    for result, path_key in [
        (diagnostics_result, "diagnostics_path"),
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
        "output_path": latest_output_path,
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
            "estimated_rates": estimated_rates_result,
            "diagnostics": diagnostics_result,
            "decision_report": report_result,
        },
    }
