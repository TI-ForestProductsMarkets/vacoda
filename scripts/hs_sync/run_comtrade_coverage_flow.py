"""
VACODA HS-sync coverage flow.

This flow audits how observed Comtrade records from multiple HS systems can be
expressed in a target HS system, using normalized UNSD/Comtrade HS
correspondence tables.

It does not run the Harvard/Growth Lab generator.
It does not generate empirical split weights.
"""

from __future__ import annotations
from datetime import datetime

from typing import Any

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.hs_sync.comtrade_mapping_audit import run_comtrade_mapping_audit
from vacoda.hs_sync.coverage_audit import run_target_hs_mapping_coverage_audit
from vacoda.utils.logger import get_logger
from vacoda.config.paths import resolve_log_path

from vacoda.utils.timing import timed_step, timed_workflow
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)

logger = get_logger("HS_SYNC.COVERAGE_FLOW")


def run_hs_sync_coverage_flow(
    *,
    cfg: HsSyncConfig,
    run_id: str,
) -> dict[str, Any]:
    """
    Run the Comtrade-only HS-sync coverage audit.

    This workflow uses normalized UNSD/Comtrade HS correspondence files already
    prepared under references/comtrade/hs_correlation_groups/normalized.
    """

    outputs: dict[str, Any] = {
        "run_id": run_id,
        "status": "started",
    }

    logger.info("Run ID: %s", run_id)
    logger.info("Coverage target HS: %s", cfg.coverage.target_hs)
    logger.info(
        "Coverage year range: %s–%s",
        cfg.coverage.min_year,
        cfg.coverage.max_year,
    )
    logger.info("Coverage mapping source: UNSD/Comtrade normalized mappings")

    with timed_workflow(logger, "HS-sync coverage flow"):

        with timed_step(
            logger,
            1,
            "Audit target-HS mapping coverage",
            total_steps=1):
            coverage_outputs = run_target_hs_mapping_coverage_audit(
                cfg=cfg,
                run_id=run_id,
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

        
        
        with timed_step(
                    logger,
                    2,
                    "Audit normalized Comtrade mappings and coverage invariants",
                    total_steps=2):
                    
                    audit_outputs = run_comtrade_mapping_audit(
                        cfg=cfg,
                        run_id=run_id,
                        expect_no_missing=True,
                        fail_on_error=True,
                    )

                    outputs["comtrade_mapping_audit"] = audit_outputs

                    logger.info("Comtrade mapping audit complete.")
                    logger.info("  Report latest: %s", audit_outputs.get("report_latest"))

        
        outputs["status"] = "completed"

    return outputs

def main() -> None:
    cfg = HsSyncConfig.from_yaml()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(cfg.logging.level)
    log_path = resolve_log_path(
        cfg.logging.resolve_log_file(),
        timestamp=cfg.logging.timestamp,
    )

    setup_logging(level=log_level, log_file=log_path)

    logger = get_logger("SCRIPT.HS_SYNC_COVERAGE_FLOW")

    try:
        # Run the coverage flow
        run_outputs = run_hs_sync_coverage_flow(
            cfg=cfg,
            run_id=run_id,
        )
    except Exception:
        logger.exception("HS-sync coverage flow failed.")
        logger.info("Log file: %s", log_path)
        raise

    logger.info("")
    logger.info("HS-sync coverage flow finished.")
    logger.info("  Status: %s", run_outputs.get("status"))
    logger.info("  Run ID: %s", run_outputs.get("run_id"))
    logger.info("  Log file: %s", log_path)

    coverage_outputs = run_outputs.get("target_hs_mapping_coverage", {})
    logger.info(
        "  Final coverage artifact: %s",
        coverage_outputs.get("coverage_report_latest"),
    )


if __name__ == "__main__":
    main()