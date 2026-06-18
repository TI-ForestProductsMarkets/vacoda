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
from vacoda.cif_fob.training_features import build_training_features
from vacoda.cif_fob.training_sample import build_training_sample
from vacoda.config.cif_fob import CifFobConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.logger import get_logger


WORKFLOW_STEPS = [
    "STEP 1 — Prepare CIF/FOB input",
    "STEP 2 — Build CIF/FOB mirror pairs",
    "STEP 3 — Build CIF/FOB training sample",
    "STEP 4 — Build CIF/FOB training features",
    "STEP 5 — Estimate CIF/FOB correction rates",
]


def _artifact_result(
    *,
    path_key: str,
    path: Path,
) -> dict[str, Any]:
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
    producer_product: str,
) -> Path:
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
    title: str,
) -> None:
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
        "training_features=%s estimated_rates=%s corrected_trade=%s",
        cfg.products.prepared_input.enable,
        cfg.products.mirror_pairs.enable,
        cfg.products.training_sample.enable,
        cfg.products.training_features.enable,
        cfg.products.estimated_rates.enable,
        cfg.products.corrected_trade.enable,
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

    # ==================================================================
    # STEP 1 — Prepare canonical CIF/FOB input
    # ==================================================================

    if cfg.products.prepared_input.enable:
        _log_step_header(
            logger=logger,
            symbol=build,
            title=WORKFLOW_STEPS[0],
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
            WORKFLOW_STEPS[0],
        )

    # ==================================================================
    # STEP 2 — Build bilateral mirror pairs
    # ==================================================================

    if cfg.products.mirror_pairs.enable:
        _log_step_header(
            logger=logger,
            symbol=build,
            title=WORKFLOW_STEPS[1],
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
            WORKFLOW_STEPS[1],
        )

    # ==================================================================
    # STEP 3 — Build factor-candidate training sample
    # ==================================================================

    if cfg.products.training_sample.enable:
        _log_step_header(
            logger=logger,
            symbol=build,
            title=WORKFLOW_STEPS[2],
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
            WORKFLOW_STEPS[2],
        )

    # ==================================================================
    # STEP 4 — Enrich training sample with predictors
    # ==================================================================

    if cfg.products.training_features.enable:
        _log_step_header(
            logger=logger,
            symbol=build,
            title=WORKFLOW_STEPS[3],
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
            WORKFLOW_STEPS[3],
        )

    # ==================================================================
    # STEP 5 — Estimate CIF/FOB correction rates
    # ==================================================================

    if cfg.products.estimated_rates.enable:
        _log_step_header(
            logger=logger,
            symbol=build,
            title=WORKFLOW_STEPS[4],
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
            WORKFLOW_STEPS[4],
        )

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
    logger.info("Latest output: %s", latest_output_path)
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
        "products_enabled": {
            "prepared_input": cfg.products.prepared_input.enable,
            "mirror_pairs": cfg.products.mirror_pairs.enable,
            "training_sample": cfg.products.training_sample.enable,
            "training_features": cfg.products.training_features.enable,
            "estimated_rates": cfg.products.estimated_rates.enable,
            "corrected_trade": cfg.products.corrected_trade.enable,
        },
        "steps": {
            "prepared_input": prepared_result,
            "mirror_pairs": mirror_result,
            "training_sample": training_sample_result,
            "training_features": training_features_result,
            "estimated_rates": estimated_rates_result,
        },
    }
