"""
VACODA HS-sync coverage/audit flow.

This flow runs the diagnostic HS-sync preparation and audit sequence:

1. Stage Comtrade data for the Harvard/Growth Lab generator.
2. Audit staged Harvard-generator inputs / frozen Harvard reference artifacts.
3. Audit Harvard mapping direction.
4. Audit target-HS mapping coverage.

This flow does not apply HS harmonization weights.
"""

from __future__ import annotations

from typing import Any

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.hs_sync.coverage_audit import run_target_hs_mapping_coverage_audit
from vacoda.hs_sync.harvard.harvard_reference_audit import run_harvard_generator_audit_from_config
from vacoda.hs_sync.harvard.harvard_stage import run_harvard_staging
from vacoda.hs_sync.harvard.mapping_direction import run_harvard_mapping_direction_audit_from_config
from vacoda.utils.logger import get_logger
from vacoda.utils.timing import timed_step, timed_workflow


logger = get_logger("HS_SYNC.COVERAGE_FLOW")


def run_hs_sync_coverage_flow(
    *,
    cfg: HsSyncConfig,
    run_id: str,
) -> dict[str, Any]:
    """
    Run the full HS-sync coverage/audit flow.

    This is the workflow-level entry point for the coverage stage.
    Step-level scripts may still call the individual backend functions.
    """

    outputs: dict[str, Any] = {
        "run_id": run_id,
        "status": "started",
    }

    logger.info("=" * 72)
    logger.info("Starting HS-sync coverage flow.")
    logger.info("=" * 72)
    logger.info("Run ID: %s", run_id)
    logger.info("Coverage target HS: %s", cfg.coverage.target_hs)
    logger.info(
        "Coverage year range: %s–%s",
        cfg.coverage.min_year,
        cfg.coverage.max_year,
    )

    with timed_workflow(logger, "HS-sync coverage flow"):

        # --------------------------------------------------------------
        # Step 1: Stage Comtrade data for Harvard generator
        # --------------------------------------------------------------
        with timed_step(
            logger,
            1,
            "Stage Comtrade data for Harvard generator",
            total_steps=4,
        ):
            staging_outputs = run_harvard_staging(
                cfg=cfg,
                run_id=run_id,
            )

            outputs["harvard_staging"] = staging_outputs

            logger.info("Harvard staging complete.")
            logger.info(
                "  Status: %s",
                staging_outputs.get("status", "completed"),
            )
            logger.info(
                "  Output directory: %s",
                staging_outputs.get("output_dir"),
            )
            logger.info(
                "  Artifact directory: %s",
                staging_outputs.get("artifact_dir"),
            )

            if staging_outputs.get("marker_path") is not None:
                logger.info(
                    "  Success marker: %s",
                    staging_outputs.get("marker_path"),
                )

            if staging_outputs.get("skip_reason") is not None:
                logger.info(
                    "  Skip reason: %s",
                    staging_outputs.get("skip_reason"),
                )

        # --------------------------------------------------------------
        # Step 2: Audit Harvard generator inputs / frozen artifacts
        # --------------------------------------------------------------
        with timed_step(
            logger,
            2,
            "Audit Harvard generator inputs",
            total_steps=4,
        ):
            generator_audit_outputs = run_harvard_generator_audit_from_config(cfg)

            outputs["harvard_generator_audit"] = generator_audit_outputs

            if generator_audit_outputs.get("status") == "disabled":
                logger.info("Harvard generator audit disabled in config; skipping.")
            else:
                logger.info("Harvard generator audit complete.")
                logger.info(
                    "  Artifact directory: %s",
                    generator_audit_outputs.get("artifact_dir"),
                )
                logger.info(
                    "  Artifact: %s",
                    generator_audit_outputs.get("artifact"),
                )

                if generator_audit_outputs.get("artifact_latest") is not None:
                    logger.info(
                        "  Artifact latest: %s",
                        generator_audit_outputs.get("artifact_latest"),
                    )

        # --------------------------------------------------------------
        # Step 3: Audit Harvard mapping direction
        # --------------------------------------------------------------
        with timed_step(
            logger,
            3,
            "Audit Harvard mapping direction",
            total_steps=4,
        ):
            mapping_direction_outputs = (
                run_harvard_mapping_direction_audit_from_config(
                    cfg=cfg,
                    logger=logger,
                )
            )

            outputs["harvard_mapping_direction"] = mapping_direction_outputs

            if mapping_direction_outputs.get("status") == "disabled":
                logger.info(
                    "Harvard mapping-direction audit disabled in config; skipping."
                )
            else:
                logger.info("Harvard mapping-direction audit complete.")
                logger.info(
                    "  Report directory: %s",
                    mapping_direction_outputs.get("report_dir"),
                )

                if mapping_direction_outputs.get("report_latest") is not None:
                    logger.info(
                        "  Report latest: %s",
                        mapping_direction_outputs.get("report_latest"),
                    )

        # --------------------------------------------------------------
        # Step 4: Audit target-HS mapping coverage
        # --------------------------------------------------------------
        with timed_step(
            logger,
            4,
            "Audit target-HS mapping coverage",
            total_steps=4,
        ):
            coverage_outputs = run_target_hs_mapping_coverage_audit(
                cfg=cfg,
                run_id=run_id,
                logger=logger,
            )

            outputs["target_hs_mapping_coverage"] = coverage_outputs

            logger.info("Target-HS mapping coverage audit complete.")
            logger.info(
                "  Output directory: %s",
                coverage_outputs.get("output_dir"),
            )
            logger.info(
                "  Artifact directory: %s",
                coverage_outputs.get("artifact_dir"),
            )

            if coverage_outputs.get("coverage_report_latest") is not None:
                logger.info(
                    "  Coverage report latest: %s",
                    coverage_outputs.get("coverage_report_latest"),
                )

        outputs["status"] = "completed"

    logger.info("=" * 72)
    logger.info("HS-sync coverage flow complete.")
    logger.info("=" * 72)

    return outputs