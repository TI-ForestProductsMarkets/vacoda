#!/usr/bin/env python3
"""
Run the VACODA master pipeline.

Current supported steps:
    1. staging
    2. hs_sync
    3. (future) cif_fob_sync
    4. (future) trade_sync
    5. (future) quantity_sync
    6. (future) value_sync
    7. (future) validation

Each module keeps its own config and audit logic.
The pipeline only orchestrates steps and checks status/ready flags.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from vacoda.config.pipeline import PipelineConfig
from vacoda.config.dataset import DatasetConfig
from vacoda.config.hs_sync import HsSyncConfig

from vacoda.dataset.staging import build_comtrade_staging_dataset
from vacoda.hs_sync.workflow import run_hs_sync_workflow
from vacoda.config.paths import resolve_log_path
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)


def _run_staging_step(
    *,
    config_path: Path,
    run_id: str,
    logger,
) -> dict[str, Any]:
    cfg = DatasetConfig.from_yaml(config_path)

    return build_comtrade_staging_dataset(
        cfg=cfg,
        run_id=run_id,
        logger=get_logger("PIPELINE.STAGING"),
    )


def _run_hs_sync_step(
    *,
    config_path: Path,
    run_id: str,
    logger,
) -> dict[str, Any]:
    cfg = HsSyncConfig.from_yaml(config_path)

    return run_hs_sync_workflow(
        cfg=cfg,
        run_id=run_id,
        logger=get_logger("PIPELINE.HS_SYNC"),
    )


def _validate_step_result(
    *,
    step_name: str,
    result: dict[str, Any],
    expected_status: list[str],
    required_ready_flag: str | None,
) -> tuple[bool, str]:
    status = result.get("status")

    if status not in expected_status:
        return (
            False,
            (
                f"Step {step_name!r} returned status={status!r}, "
                f"expected one of {expected_status!r}."
            ),
        )

    if required_ready_flag is not None:
        ready_value = result.get(required_ready_flag)

        if ready_value is not True:
            return (
                False,
                (
                    f"Step {step_name!r} did not satisfy required ready flag "
                    f"{required_ready_flag!r}. Actual value: {ready_value!r}."
                ),
            )

    return True, ""


def _manifest_row(
    *,
    run_id: str,
    step_name: str,
    status: str,
    ready_flag: str | None,
    ready_value: Any,
    duration_seconds: float,
    result: dict[str, Any],
    message: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "step_name": step_name,
        "status": status,
        "required_ready_flag": ready_flag,
        "ready_value": ready_value,
        "duration_seconds": round(duration_seconds, 3),
        "output_path": str(
            result.get("output_path")
            or result.get("staged_parquet")
            or result.get("final_parquet")
            or ""
        ),
        "audit_path": str(
            result.get("audit_path")
            or result.get("audit_file")
            or result.get("final_audit")
            or ""
        ),
        "n_failed_audit_rows": result.get("n_failed_audit_rows"),
        "message": message,
    }


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = PipelineConfig.from_yaml("configs/pipeline.yaml")

    log_level = parse_log_level(cfg.logging.level)
    log_path = resolve_log_path(
        cfg.logging.resolve_log_file(),
        timestamp=cfg.logging.timestamp,
    )

    setup_logging(level=log_level, log_file=log_path)
    logger = get_logger("PIPELINE")

    manifest_path = cfg.manifest.resolve_path()
    if cfg.manifest.timestamp:
        manifest_path = manifest_path.with_name(
            f"{manifest_path.stem}_{run_id}{manifest_path.suffix}"
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("START — VACODA master pipeline")
    logger.info("=" * 72)
    logger.info("Run ID: %s", run_id)
    logger.info("Pipeline: %s", cfg.pipeline.name)
    logger.info("Enabled steps: %s", ", ".join(cfg.pipeline.enabled_steps))
    logger.info("Manifest: %s", manifest_path)
    logger.info("Log file: %s", log_path)

    if not cfg.pipeline.enabled:
        logger.warning("Pipeline is disabled in config. Nothing to run.")
        return

    step_runners = {
        "staging": _run_staging_step,
        "hs_sync": _run_hs_sync_step,
    }

    manifest_rows: list[dict[str, Any]] = []
    pipeline_status = "completed"

    for step_name, step_cfg in cfg.enabled_steps():
        logger.info("=" * 72)
        logger.info("STEP — %s", step_name)
        logger.info("=" * 72)
        logger.info("Config: %s", step_cfg.config)
        logger.info("Expected status: %s", step_cfg.expected_status)
        logger.info("Required ready flag: %s", step_cfg.required_ready_flag)

        if step_name not in step_runners:
            message = f"No runner implemented for pipeline step {step_name!r}."
            logger.error(message)

            manifest_rows.append(
                _manifest_row(
                    run_id=run_id,
                    step_name=step_name,
                    status="failed",
                    ready_flag=step_cfg.required_ready_flag,
                    ready_value=None,
                    duration_seconds=0.0,
                    result={},
                    message=message,
                )
            )

            pipeline_status = "failed"
            if cfg.pipeline.fail_fast:
                break
            continue

        start = perf_counter()

        try:
            result = step_runners[step_name](
                config_path=step_cfg.config,
                run_id=run_id,
                logger=logger,
            )
            duration_seconds = perf_counter() - start

            ok, message = _validate_step_result(
                step_name=step_name,
                result=result,
                expected_status=step_cfg.expected_status,
                required_ready_flag=step_cfg.required_ready_flag,
            )

            status = str(result.get("status", "unknown"))
            ready_value = (
                None
                if step_cfg.required_ready_flag is None
                else result.get(step_cfg.required_ready_flag)
            )

            if ok:
                logger.info("DONE — %s (%.2fs)", step_name, duration_seconds)
                logger.info("  Status: %s", status)
                logger.info("  Ready flag %s: %s", step_cfg.required_ready_flag, ready_value)
            else:
                logger.error("FAILED — %s", step_name)
                logger.error("  %s", message)
                pipeline_status = "failed"

            manifest_rows.append(
                _manifest_row(
                    run_id=run_id,
                    step_name=step_name,
                    status=status if ok else "failed",
                    ready_flag=step_cfg.required_ready_flag,
                    ready_value=ready_value,
                    duration_seconds=duration_seconds,
                    result=result,
                    message=message,
                )
            )

            if not ok and cfg.pipeline.fail_fast:
                break

        except Exception as exc:
            duration_seconds = perf_counter() - start
            pipeline_status = "failed"

            logger.exception("Step failed: %s", step_name)

            manifest_rows.append(
                _manifest_row(
                    run_id=run_id,
                    step_name=step_name,
                    status="failed",
                    ready_flag=step_cfg.required_ready_flag,
                    ready_value=None,
                    duration_seconds=duration_seconds,
                    result={},
                    message=str(exc),
                )
            )

            if cfg.pipeline.fail_fast:
                break

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)

    logger.info("=" * 72)
    logger.info("VACODA master pipeline finished.")
    logger.info("Status: %s", pipeline_status)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Log file: %s", log_path)
    logger.info("=" * 72)

    if pipeline_status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()