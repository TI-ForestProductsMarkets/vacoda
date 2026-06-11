from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from vacoda.config.dataset import DatasetConfig
from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.paths import resolve_log_path
from vacoda.config.pipeline import PipelineConfig
from vacoda.dataset.staging import build_comtrade_staging_dataset
from vacoda.hs_sync.workflow import run_hs_sync_workflow
from vacoda.pipeline.checkpoint import (
    checkpoint_path,
    config_fingerprint,
    file_list_fingerprint,
    file_manifest_fingerprint,
    load_checkpoint,
    validate_checkpoint,
    write_checkpoint,
)
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)


@dataclass(frozen=True)
class StepPlan:
    step_name: str
    config_path: Path
    config_fingerprint: dict[str, Any]
    input_fingerprint: dict[str, Any]
    output_path: Path
    audit_path: Path
    runner: Callable[[str], dict[str, Any]]


def _run_staging_step(
    *,
    config_path: Path,
    run_id: str,
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
) -> dict[str, Any]:
    cfg = HsSyncConfig.from_yaml(config_path)

    return run_hs_sync_workflow(
        cfg=cfg,
        run_id=run_id,
        logger=get_logger("PIPELINE.HS_SYNC"),
    )


def _build_staging_plan(
    *,
    step_name: str,
    config_path: Path,
) -> StepPlan:
    cfg = DatasetConfig.from_yaml(config_path)

    if cfg.input.pattern is None:
        raise ValueError("dataset input.pattern is required for staging checkpointing.")

    input_dir = cfg.input.resolve_dir()
    input_pattern = cfg.input.pattern

    return StepPlan(
        step_name=step_name,
        config_path=config_path,
        config_fingerprint=config_fingerprint(config_path),
        input_fingerprint=file_manifest_fingerprint(
            root=input_dir,
            pattern=input_pattern,
        ),
        output_path=cfg.output_path(),
        audit_path=cfg.audit_path(),
        runner=lambda run_id: _run_staging_step(
            config_path=config_path,
            run_id=run_id,
        ),
    )


def _build_hs_sync_plan(
    *,
    step_name: str,
    config_path: Path,
) -> StepPlan:
    cfg = HsSyncConfig.from_yaml(config_path)

    input_dir = cfg.apply_trade_data_dir()
    input_pattern = cfg.coverage.input.pattern

    if input_pattern is None:
        raise ValueError("coverage.input.pattern is required for HS-sync checkpointing.")

    # HS-sync depends on:
    #   1. staged trade input
    #   2. hs_sync.yaml config
    #
    # Mapping/reference files are currently covered indirectly by hs_sync.yaml
    # and by the final audit. Later, add a mapping file-tree fingerprint here
    # if your HsSyncConfig exposes the normalized mapping directory cleanly.
    mapping_files = cfg.raw_mapping_file_paths()

    if not mapping_files:
        raise ValueError("No raw HS mapping files found in hs_sync config.")

    missing = [path for path in mapping_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
        "Missing configured raw HS mapping files:\n"
        + "\n".join(str(path) for path in missing)
    )

    input_fp = {
        "trade_input": file_manifest_fingerprint(
            root=input_dir,
            pattern=input_pattern,
        ),
        "mapping_files": file_list_fingerprint(
            paths=mapping_files,
            label="configured_raw_mapping_files",
        ),
    }

    return StepPlan(
        step_name=step_name,
        config_path=config_path,
        config_fingerprint=config_fingerprint(config_path),
        input_fingerprint=input_fp,
        output_path=cfg.apply_output_path(),
        audit_path=cfg.apply_integrity_audit_path(),
        runner=lambda run_id: _run_hs_sync_step(
            config_path=config_path,
            run_id=run_id,
        ),
    )


def _build_step_plan(
    *,
    step_name: str,
    config_path: Path,
) -> StepPlan:
    if step_name == "staging":
        return _build_staging_plan(
            step_name=step_name,
            config_path=config_path,
        )

    if step_name == "hs_sync":
        return _build_hs_sync_plan(
            step_name=step_name,
            config_path=config_path,
        )

    raise NotImplementedError(f"No pipeline plan implemented for step {step_name!r}.")


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
    output_path: Path | str | None,
    audit_path: Path | str | None,
    n_failed_audit_rows: Any,
    message: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "step_name": step_name,
        "status": status,
        "required_ready_flag": ready_flag,
        "ready_value": ready_value,
        "duration_seconds": round(duration_seconds, 3),
        "output_path": "" if output_path is None else str(output_path),
        "audit_path": "" if audit_path is None else str(audit_path),
        "n_failed_audit_rows": n_failed_audit_rows,
        "message": message,
    }


def _result_output_path(result: dict[str, Any], fallback: Path) -> Path:
    value = (
        result.get("output_path")
        or result.get("staged_parquet")
        or result.get("final_parquet")
        or result.get("applied_path")
        or fallback
    )

    return Path(value)


def _result_audit_path(result: dict[str, Any], fallback: Path) -> Path:
    value = (
        result.get("audit_path")
        or result.get("audit_file")
        or result.get("final_audit")
        or fallback
    )

    return Path(value)


def run_pipeline(
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = PipelineConfig.from_yaml()

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

    checkpoint_dir = cfg.checkpoint.resolve_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("START — VACODA master pipeline")
    logger.info("=" * 72)
    logger.info("Run ID: %s", run_id)
    logger.info("Pipeline: %s", cfg.pipeline.name)
    logger.info("Enabled steps: %s", ", ".join(cfg.pipeline.enabled_steps))
    logger.info("Checkpoint reuse: %s", cfg.checkpoint.reuse_successful_steps)
    logger.info("Force run all: %s", cfg.pipeline.force_run_all)
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Log file: %s", log_path)

    if cfg.pipeline.force_run_all:
        logger.warning("FORCE RUN ALL is enabled — valid checkpoints will be ignored.")

    if not cfg.pipeline.enabled:
        logger.warning("Pipeline is disabled in config. Nothing to run.")
        return {
            "status": "disabled",
            "run_id": run_id,
            "manifest_path": manifest_path,
            "log_path": log_path,
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

        try:
            plan = _build_step_plan(
                step_name=step_name,
                config_path=step_cfg.config,
            )
        except Exception as exc:
            message = f"Could not build pipeline plan for {step_name!r}: {exc}"
            logger.exception(message)

            manifest_rows.append(
                _manifest_row(
                    run_id=run_id,
                    step_name=step_name,
                    status="failed",
                    ready_flag=step_cfg.required_ready_flag,
                    ready_value=None,
                    duration_seconds=0.0,
                    output_path=None,
                    audit_path=None,
                    n_failed_audit_rows=None,
                    message=message,
                )
            )

            pipeline_status = "failed"
            if cfg.pipeline.fail_fast:
                break
            continue

        step_checkpoint_path = checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            step_name=step_name,
        )

        if cfg.checkpoint.enabled and cfg.checkpoint.reuse_successful_steps and not cfg.pipeline.force_run_all:
            checkpoint = load_checkpoint(step_checkpoint_path)

            if checkpoint is not None:
                reusable, reason = validate_checkpoint(
                    checkpoint=checkpoint,
                    expected_status=step_cfg.expected_status,
                    required_ready_flag=step_cfg.required_ready_flag,
                    current_config=plan.config_fingerprint,
                    current_input=plan.input_fingerprint,
                )

                if reusable:
                    logger.info("SKIP — %s", step_name)
                    logger.info("  Reason: valid checkpoint exists.")
                    logger.info("  Checkpoint: %s", step_checkpoint_path)
                    logger.info("  Output: %s", checkpoint.get("output", {}).get("path"))
                    logger.info("  Audit: %s", checkpoint.get("audit", {}).get("path"))

                    manifest_rows.append(
                        _manifest_row(
                            run_id=run_id,
                            step_name=step_name,
                            status="skipped_checkpoint",
                            ready_flag=step_cfg.required_ready_flag,
                            ready_value=checkpoint.get("ready_value"),
                            duration_seconds=0.0,
                            output_path=checkpoint.get("output", {}).get("path"),
                            audit_path=checkpoint.get("audit", {}).get("path"),
                            n_failed_audit_rows=checkpoint.get("audit", {}).get(
                                "n_failed_audit_rows"
                            ),
                            message="Reused valid previous checkpoint.",
                        )
                    )

                    continue

                logger.info("Checkpoint exists but is not reusable.")
                logger.info("  Checkpoint: %s", step_checkpoint_path)
                logger.info("  old: %s", checkpoint.get("input"))
                logger.info("  new: %s", plan.input_fingerprint)
                logger.info("  Reason: %s", reason)

        start = perf_counter()

        try:
            result = plan.runner(run_id)
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

            output_path = _result_output_path(result, plan.output_path)
            audit_path = _result_audit_path(result, plan.audit_path)

            if ok:
                logger.info("DONE — %s (%.2fs)", step_name, duration_seconds)
                logger.info("  Status: %s", status)
                logger.info("  Ready flag %s: %s", step_cfg.required_ready_flag, ready_value)

                if cfg.checkpoint.enabled:
                    write_checkpoint(
                        path=step_checkpoint_path,
                        run_id=run_id,
                        step_name=step_name,
                        status=status,
                        ready_flag=step_cfg.required_ready_flag,
                        ready_value=ready_value,
                        config=plan.config_fingerprint,
                        input_fingerprint=plan.input_fingerprint,
                        output_path=output_path,
                        audit_path=audit_path,
                        result=result,
                    )

                    logger.info("  Checkpoint written: %s", step_checkpoint_path)

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
                    output_path=output_path,
                    audit_path=audit_path,
                    n_failed_audit_rows=result.get("n_failed_audit_rows"),
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
                    output_path=plan.output_path,
                    audit_path=plan.audit_path,
                    n_failed_audit_rows=None,
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

    return {
        "status": pipeline_status,
        "run_id": run_id,
        "manifest_path": manifest_path,
        "log_path": log_path,
        "n_steps": len(manifest_rows),
    }
