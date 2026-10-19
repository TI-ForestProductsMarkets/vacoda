"""vacoda.steps.release.workflow

Orchestrate VACODA Release after Validation.

This module contains orchestration only. Candidate construction, audits,
publication, diagnostics, and reporting belong to their respective modules.

NOTE: The five imported Release step functions are interfaces to implement next.
This workflow cannot run until those modules exist and honor the contracts below.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from vacoda.config.steps.release import ReleaseConfig
from vacoda.utils.artifacts import require_artifact, require_enabled_products
from vacoda.utils.display import get_symbols
from vacoda.utils.runtime import run_step, separator, timed_workflow

from vacoda.steps.release.release_data import build_release_candidates
from vacoda.steps.release.audit_release import audit_release_candidates
from vacoda.steps.release.publish import publish_release_datasets
from vacoda.steps.release.diagnostics import build_release_diagnostics
from vacoda.steps.release.reporting import build_release_report


REQUIRED_PRODUCTS = (
    "release",
    "release_audit",
    "diagnostics",
)

WORKFLOW_STEPS = [
    "Build Release datasets",
    "Audit Release datasets",
    "Publish Release datasets",
    "Build Release diagnostics",
    "Write Release decision report",
]

TOTAL_WORKFLOW_STEPS = len(WORKFLOW_STEPS)


def run_release_workflow(
    *,
    cfg: ReleaseConfig,
    run_id: str,
    release_id: str,
    logger,
    force: bool = False,
    prepared_input_path: Path | None = None,
    inference_input_path: Path | None = None,
    training_definition: str | None = None,
) -> dict[str, Any]:
    """Release a validated canonical population as standard/enriched datasets.

    Pipeline callers should supply BOTH resolved Validation artifact paths.
    Standalone callers may omit prepared_input_path; to resolve inference
    independently, supply training_definition if inference_input_path is absent.

    release_id identifies an immutable configuration-specific publication
    directory; the master pipeline must choose and supply that identity.
    """
    prepared_path = (
        Path(prepared_input_path)
        if prepared_input_path is not None
        else cfg.prepared_input_path()
    )

    if inference_input_path is not None:
        inference_path = Path(inference_input_path)
    elif training_definition is not None:
        inference_path = cfg.inference_input_path(
            training_definition=training_definition
        )
    else:
        raise ValueError(
            "Release requires inference_input_path or training_definition."
        )

    for label, path in (
        ("Validation Prepared", prepared_path),
        ("Validation Inference", inference_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} parquet not found: {path}")

    enabled_products = cfg.enabled_products()
    require_enabled_products(
        workflow_name="Release",
        enabled_products=enabled_products,
        required_products=REQUIRED_PRODUCTS,
    )

    # Resolve and validate identity without creating a published directory.
    standard_path = cfg.product_path("release", release_id=release_id)
    enriched_path = cfg.enriched_path(release_id=release_id)
    staging_dir = cfg.output_dir() / ".staging" / release_id

    workflow_start = perf_counter()

    with timed_workflow(logger, "Release"):
        logger.info("Run ID: %s", run_id)
        logger.info("Release identity: %s", release_id)
        logger.info("Target HS: %s", cfg.scope.target_hs)
        logger.info("Year range: %s-%s", cfg.scope.min_year, cfg.scope.max_year)
        logger.info("Prepared input: %s", prepared_path)
        logger.info("Inference input: %s", inference_path)
        logger.info("Staging directory: %s", staging_dir)
        logger.info("Publication directory: %s", standard_path.parent)
        logger.info(
            "Enabled products: %s",
            " ".join(f"{name}={enabled}" for name, enabled in enabled_products.items()),
        )

        # ==================================================================
        # Step 1 — Build both candidate datasets (staging only)
        # ==================================================================
        build_result = run_step(
            logger=logger,
            step_number=1,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[0],
            runner=lambda: build_release_candidates(
                cfg=cfg,
                prepared_input_path=prepared_path,
                inference_input_path=inference_path,
                staging_dir=staging_dir,
                run_id=run_id,
                force=force,
                logger=logger,
            ),
        )
        candidate_standard_path = require_artifact(
            result=build_result,
            path_key="candidate_standard_path",
            consumer_step=WORKFLOW_STEPS[1],
            producer_product="standard Release candidate",
        )
        candidate_enriched_path = require_artifact(
            result=build_result,
            path_key="candidate_enriched_path",
            consumer_step=WORKFLOW_STEPS[1],
            producer_product="enriched Release candidate",
        )

        # ==================================================================
        # Step 2 — Audit candidates (must pass before publication)
        # ==================================================================
        audit_result = run_step(
            logger=logger,
            step_number=2,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[1],
            runner=lambda: audit_release_candidates(
                cfg=cfg,
                prepared_input_path=prepared_path,
                inference_input_path=inference_path,
                candidate_standard_path=candidate_standard_path,
                candidate_enriched_path=candidate_enriched_path,
                run_id=run_id,
                release_id=release_id,
                logger=logger,
            ),
        )
        audit_path = require_artifact(
            result=audit_result,
            path_key="audit_path",
            consumer_step=WORKFLOW_STEPS[2],
            producer_product="Release candidate audit",
        )
        if audit_result.get("audit_passed") is not True:
            raise RuntimeError("Release candidate audit did not pass; publication blocked.")

        # ==================================================================
        # Step 3 — Publish ONLY the audited candidates and their metadata
        # ==================================================================
        publish_result = run_step(
            logger=logger,
            step_number=3,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[2],
            runner=lambda: publish_release_datasets(
                cfg=cfg,
                candidate_standard_path=candidate_standard_path,
                candidate_enriched_path=candidate_enriched_path,
                audit_path=audit_path,
                standard_path=standard_path,
                enriched_path=enriched_path,
                run_id=run_id,
                release_id=release_id,
                force=force,
                logger=logger,
            ),
        )
        published_standard_path = require_artifact(
            result=publish_result,
            path_key="standard_path",
            consumer_step=WORKFLOW_STEPS[3],
            producer_product="published standard dataset",
        )
        published_enriched_path = require_artifact(
            result=publish_result,
            path_key="enriched_path",
            consumer_step=WORKFLOW_STEPS[3],
            producer_product="published enriched dataset",
        )

        # ==================================================================
        # Step 4 — Diagnostics (informational; never authorizes publication)
        # ==================================================================
        diagnostics_result = run_step(
            logger=logger,
            step_number=4,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[3],
            runner=lambda: build_release_diagnostics(
                cfg=cfg,
                standard_path=published_standard_path,
                enriched_path=published_enriched_path,
                audit_path=audit_path,
                run_id=run_id,
                release_id=release_id,
                force=force,
                logger=logger,
            ),
        )
        diagnostics_path = require_artifact(
            result=diagnostics_result,
            path_key="diagnostics_path",
            consumer_step=WORKFLOW_STEPS[4],
            producer_product="Release diagnostics",
        )

        # ==================================================================
        # Step 5 — Write Release report (descriptive, not approval authority)
        # ==================================================================
        report_result = run_step(
            logger=logger,
            step_number=5,
            total_steps=TOTAL_WORKFLOW_STEPS,
            title=WORKFLOW_STEPS[4],
            runner=lambda: build_release_report(
                cfg=cfg,
                standard_path=published_standard_path,
                enriched_path=published_enriched_path,
                audit_path=audit_path,
                diagnostics_path=diagnostics_path,
                run_id=run_id,
                release_id=release_id,
                force=force,
                logger=logger,
            ),
        )
        report_path = require_artifact(
            result=report_result,
            path_key="report_path",
            consumer_step="Release workflow completion",
            producer_product="Release decision report",
        )

        # Audit gating happened BEFORE publication. Completion additionally
        # requires both published artifacts, diagnostics, and report.
        release_completed = all(
            path.is_file()
            for path in (
                published_standard_path,
                published_enriched_path,
                audit_path,
                diagnostics_path,
                report_path,
            )
        )
        if not release_completed:
            raise RuntimeError("Release finished without complete publication artifacts.")

        workflow_elapsed = perf_counter() - workflow_start
        separator(logger)
        logger.info("%s Release workflow finished.", get_symbols("success"))
        logger.info("Standard dataset: %s", published_standard_path)
        logger.info("Enriched dataset: %s", published_enriched_path)
        logger.info("Audit: %s", audit_path)
        logger.info("Diagnostics: %s", diagnostics_path)
        logger.info("Report: %s", report_path)
        logger.info("Release completed: %s", release_completed)
        logger.info("Workflow duration: %.2fs", workflow_elapsed)
        separator(logger)

    return {
        "status": "completed",
        "run_id": run_id,
        "release_id": release_id,
        "release_completed": release_completed,
        "output_path": published_standard_path,
        "standard_path": published_standard_path,
        "enriched_path": published_enriched_path,
        "audit_path": audit_path,
        "diagnostics_path": diagnostics_path,
        "report_path": report_path,
        "latest_report_path": report_result.get("latest_report_path"),
        "audit_passed": True,
        "workflow_duration_seconds": workflow_elapsed,
    }
