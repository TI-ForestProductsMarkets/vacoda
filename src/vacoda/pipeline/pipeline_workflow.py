"""
vacoda.pipeline.pipeline_workflow
-----------------------------

VACODA master pipeline implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from vacoda.config.steps.trade_sync import TradeSyncConfig
from vacoda.config.steps.cif_fob import CifFobConfig
from vacoda.config.dataset import DatasetConfig
from vacoda.config.steps.hs_sync import HsSyncConfig
from vacoda.config.steps.value import ValueConfig
from vacoda.config.paths import resolve_log_path
from vacoda.config.pipeline import PipelineConfig
from vacoda.steps.trade_sync.workflow import run_trade_sync_workflow
from vacoda.config.steps.quantity import QuantityConfig
from vacoda.steps.quantity.audit import quantity_audit_path
from vacoda.steps.quantity.workflow import run_quantity_workflow
from vacoda.steps.cif_fob.workflow import run_cif_fob_workflow
from vacoda.steps.dataset.staging import build_comtrade_staging_dataset
from vacoda.steps.hs_sync.workflow import run_hs_sync_workflow
from vacoda.steps.value.workflow import run_value_workflow
from vacoda.pipeline.checkpoint import (
    checkpoint_path,
    config_fingerprint,
    semantic_config_fingerprint,
    file_list_fingerprint,
    file_manifest_fingerprint,
    load_checkpoint,
    validate_checkpoint,
    write_checkpoint,
    source_files_fingerprint
)
from vacoda.utils.runtime import timed_pipeline_workflow, log_pipeline_step_banner, print_preflight_step, print_workflow_banner
from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)
from vacoda.utils.display import get_symbols, color_text


@dataclass(frozen=True)
class StepPlan:
    step_name: str
    config_path: Path
    config_fingerprint: dict[str, Any]
    input_fingerprint: dict[str, Any]
    output_path: Path
    audit_path: Path
    runner: Callable[[str, bool], dict[str, Any]]

@dataclass(frozen=True)
class StepDecision:
    step_name: str
    step_config: Any
    plan: StepPlan
    checkpoint_path: Path
    action: str  # "run" or "skip"
    reason: str

def _run_staging_step(
    *,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
    run_id: str,
    force: bool = False
) -> dict[str, Any]:
    cfg = DatasetConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    return build_comtrade_staging_dataset(
        cfg=cfg,
        run_id=run_id,
        logger=get_logger("PIPELINE.STAGING"),
        force=force
    )


def _build_step_decisions(
    *,
    cfg: PipelineConfig,
    checkpoint_dir: Path,
    logger: Any,
) -> list[StepDecision]:
    decisions: list[StepDecision] = []
    upstream_will_run = False

    for step_name, step_cfg in cfg.enabled_steps():
        plan = _build_step_plan(
            step_name=step_name,
            config_path=step_cfg.config,
            pipeline_cfg=cfg,
        )

        step_checkpoint_path = checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            step_name=step_name,
        )

        action = "run"
        reason = "no reusable checkpoint"

        if upstream_will_run:
            action = "run"
            reason = "upstream pipeline step is scheduled to run"

        elif (
            cfg.checkpoint.enabled
            and cfg.checkpoint.reuse_successful_steps
            and not cfg.pipeline.force_run_all
        ):
            checkpoint = load_checkpoint(step_checkpoint_path)

            if checkpoint is not None:
                reusable, checkpoint_reason = validate_checkpoint(
                    checkpoint=checkpoint,
                    expected_status=step_cfg.expected_status,
                    required_ready_flag=step_cfg.required_ready_flag,
                    current_config=plan.config_fingerprint,
                    current_input=plan.input_fingerprint,
                )

                if reusable:
                    output_path = checkpoint.get("output", {}).get("path")
                    audit_path = checkpoint.get("audit", {}).get("path")

                    if isinstance(output_path, str) and output_path:
                        plan = StepPlan(
                            step_name=plan.step_name,
                            config_path=plan.config_path,
                            config_fingerprint=plan.config_fingerprint,
                            input_fingerprint=plan.input_fingerprint,
                            output_path=Path(output_path),
                            audit_path=(
                                Path(audit_path)
                                if isinstance(audit_path, str) and audit_path
                                else plan.audit_path
                            ),
                            runner=plan.runner,
                        )

                    action = "skip"
                    reason = "valid checkpoint exists"
                else:
                    action = "run"
                    reason = (
                        "checkpoint not reusable: "
                        f"{checkpoint_reason}"
                    )

        elif cfg.pipeline.force_run_all:
            action = "run"
            reason = "pipeline.force_run_all=true"

        elif not cfg.checkpoint.enabled:
            action = "run"
            reason = "checkpointing disabled"

        elif not cfg.checkpoint.reuse_successful_steps:
            action = "run"
            reason = "checkpoint reuse disabled"

        decisions.append(
            StepDecision(
                step_name=step_name,
                step_config=step_cfg,
                plan=plan,
                checkpoint_path=step_checkpoint_path,
                action=action,
                reason=reason,
            )
        )

        if action == "run":
            upstream_will_run = True

    return decisions

def _build_step_decisions_deprecated(
    *,
    cfg: PipelineConfig,
    checkpoint_dir: Path,
    logger: Any,) -> list[StepDecision]:
    decisions: list[StepDecision] = []

    for step_name, step_cfg in cfg.enabled_steps():
        plan = _build_step_plan(
                        step_name=step_name,
                        config_path=step_cfg.config,
                        pipeline_cfg=cfg,
                    )

        step_checkpoint_path = checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            step_name=step_name,
        )

        action = "run"
        reason = "no reusable checkpoint"

        if (
            cfg.checkpoint.enabled
            and cfg.checkpoint.reuse_successful_steps
            and not cfg.pipeline.force_run_all
        ):
            checkpoint = load_checkpoint(step_checkpoint_path)

            if checkpoint is not None:
                reusable, checkpoint_reason = validate_checkpoint(
                    checkpoint=checkpoint,
                    expected_status=step_cfg.expected_status,
                    required_ready_flag=step_cfg.required_ready_flag,
                    current_config=plan.config_fingerprint,
                    current_input=plan.input_fingerprint,
                )

                if reusable:
                    output_path = checkpoint.get("output", {}).get("path")
                    audit_path = checkpoint.get("audit", {}).get("path")

                    if isinstance(output_path, str) and output_path:
                        plan = StepPlan(
                            step_name=plan.step_name,
                            config_path=plan.config_path,
                            config_fingerprint=plan.config_fingerprint,
                            input_fingerprint=plan.input_fingerprint,
                            output_path=Path(output_path),
                            audit_path=(
                                Path(audit_path)
                                if isinstance(audit_path, str) and audit_path
                                else plan.audit_path
                            ),
                            runner=plan.runner,
                        )

                    action = "skip"
                    reason = "valid checkpoint exists"
                else:
                    action = "run"
                    reason = f"checkpoint not reusable: {checkpoint_reason}"

        elif cfg.pipeline.force_run_all:
            action = "run"
            reason = "pipeline.force_run_all=true"

        elif not cfg.checkpoint.enabled:
            action = "run"
            reason = "checkpointing disabled"

        elif not cfg.checkpoint.reuse_successful_steps:
            action = "run"
            reason = "checkpoint reuse disabled"

        decisions.append(
            StepDecision(
                step_name=step_name,
                step_config=step_cfg,
                plan=plan,
                checkpoint_path=step_checkpoint_path,
                action=action,
                reason=reason,
            )
        )

    return decisions

def _confirm_pipeline_run(
    *,
    cfg: PipelineConfig,
    run_id: str,
    manifest_path: Path,
    log_path: Path,
    checkpoint_dir: Path,
    decisions: list[StepDecision],
) -> bool:
    print_workflow_banner(
        symbol = color_text(get_symbols("play"), "red"),
        text="VACODA PIPELINE PREFLIGHT CHECK",
    )
    print(f"Run ID:              {run_id}")
    print(f"Pipeline:            {cfg.pipeline.name}")
    print(f"Enabled:             {cfg.pipeline.enabled}")
    print(f"Target HS:           {cfg.scope.target_hs}")
    print(f"Year range:          {cfg.scope.min_year} — {cfg.scope.max_year}")
    print(f"Enabled steps:       {', '.join(cfg.pipeline.enabled_steps)}")
    print(f"Engine name:         {cfg.engine.name}")
    print(f"Engine threads:      {cfg.engine.threads}")
    print(f"Fail fast:           {cfg.pipeline.fail_fast}")
    print(f"Force run all:       {cfg.pipeline.force_run_all}")
    print(f"Checkpoint enabled:  {cfg.checkpoint.enabled}")
    print(f"Checkpoint reuse:    {cfg.checkpoint.reuse_successful_steps}")
    print(f"Checkpoint dir:      {checkpoint_dir}")
    print(f"Manifest:            {manifest_path}")
    print(f"Log file:            {log_path}")
    print("-" * 72)
    print(f"{get_symbols('right_finger')} Step plan:")
    print("-" * 72)

    for decision in decisions:
        print_preflight_step(decision)

    if cfg.pipeline.force_run_all:
        print(f"{get_symbols('warning_light')} WARNING: force_run_all=true. All enabled steps will run even if valid checkpoints exist.")
        print("-" * 72)

    answer = input(f"{color_text(get_symbols('play'), 'green')}  Continue with this pipeline run? Type 'y' to continue: ").strip().lower()

    return answer == "y"

def _run_hs_sync_step(
    *,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
    run_id: str,
    force: bool = False,
) -> dict[str, Any]:
    cfg = HsSyncConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    return run_hs_sync_workflow(
        cfg=cfg,
        run_id=run_id,
        logger=get_logger("PIPELINE.HS_SYNC"),
        force=force
    )


def _run_quantity_step(
    *,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
    run_id: str,
    force: bool = False,
) -> dict[str, Any]:
    cfg = QuantityConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    return run_quantity_workflow(
        cfg=cfg,
        run_id=run_id,
        logger=get_logger("PIPELINE.QUANTITY"),
        force=force,
    )

def _run_cif_fob_step(
    *,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
    run_id: str,
    force: bool = False,
) -> dict[str, Any]:
    cfg = CifFobConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    return run_cif_fob_workflow(
        cfg=cfg,
        run_id=run_id,
        logger=get_logger("PIPELINE.CIF_FOB"),
        force=force
    )

def _run_trade_sync_step(
    *,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
    run_id: str,
    force: bool = False,
) -> dict[str, Any]:
    cfg = TradeSyncConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    return run_trade_sync_workflow(
        cfg=cfg,
        run_id=run_id,
        logger=get_logger("PIPELINE.TRADE_SYNC"),
        force=force,
    )

def _run_value_step(
    *,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
    run_id: str,
    force: bool = False,
) -> dict[str, Any]:
    cfg = ValueConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    return run_value_workflow(
        cfg=cfg,
        run_id=run_id,
        logger=get_logger("PIPELINE.VALUE"),
        force=force,
    )

def _build_staging_plan(
    *,
    step_name: str,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
) -> StepPlan:
    cfg = DatasetConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    if cfg.input.pattern is None:
        raise ValueError("dataset input.pattern is required for staging checkpointing.")

    input_dir = cfg.input.resolve_dir()
    input_pattern = cfg.input.pattern

    return StepPlan(
        step_name=step_name,
        config_path=config_path,
        config_fingerprint=_step_config_fingerprint(
            config_path=config_path,
            resolved_config=cfg,
            pipeline_cfg=pipeline_cfg,
            include_scope=False,
        ),
        input_fingerprint=file_manifest_fingerprint(
            root=input_dir,
            pattern=input_pattern,
        ),
        output_path=cfg.output_path(),
        audit_path=cfg.audit_path(),
        runner=lambda run_id, force: _run_staging_step(
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
            run_id=run_id,
            force=force
        ),
    )


def _build_hs_sync_plan(
    *,
    step_name: str,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
) -> StepPlan:

    cfg = HsSyncConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

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
        config_fingerprint=_step_config_fingerprint(
                                config_path=config_path,
                                resolved_config=cfg,
                                pipeline_cfg=pipeline_cfg,
                                include_scope=True,
                            ),
        input_fingerprint=input_fp,
        output_path=cfg.apply_output_path(),
        audit_path=cfg.apply_integrity_audit_path(),
        runner=lambda run_id, force: _run_hs_sync_step(
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
            run_id=run_id,
            force=force
        ),
    )

def _build_quantity_plan(
    *,
    step_name: str,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
) -> StepPlan:
    cfg = QuantityConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    input_path = cfg.input_path()

    quantity_source_dir = Path(
        run_quantity_workflow.__code__.co_filename
    ).parent

    implementation_fp = source_files_fingerprint(
        paths=list(
            quantity_source_dir.rglob("*.py")
        ),
        label="quantity_implementation",
    )

    input_fp = {
        "quantity_input": file_list_fingerprint(
            paths=[input_path],
            label="quantity_required_input",
        ),
        "implementation": implementation_fp,
    }

    output_path = cfg.product_path(
        "synchronized_quantity"
    )

    audit_path = quantity_audit_path(cfg)

    return StepPlan(
        step_name=step_name,
        config_path=config_path,
        config_fingerprint=_step_config_fingerprint(
            config_path=config_path,
            resolved_config=cfg,
            pipeline_cfg=pipeline_cfg,
            include_scope=True,
        ),
        input_fingerprint=input_fp,
        output_path=output_path,
        audit_path=audit_path,
        runner=lambda run_id, force: _run_quantity_step(
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
            run_id=run_id,
            force=force,
        ),
    )

def _build_cif_fob_plan(
    *,
    step_name: str,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
) -> StepPlan:
    cfg = CifFobConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    input_path = cfg.input_path()

    reference_paths = [
        cfg.features.geography.resolve_path(),
        cfg.features.pair_geography.resolve_path(),
        cfg.features.geography_crosswalk.resolve_path(),
        cfg.features.oil_price.resolve_path(),
    ]

    cif_fob_source_dir = Path(
        run_cif_fob_workflow.__code__.co_filename
    ).parent

    implementation_fp = source_files_fingerprint(
        paths=list(
            cif_fob_source_dir.rglob("*.py")
        ),
        label="cif_fob_implementation",
    )

    input_fp = {
        "cif_fob_input": file_list_fingerprint(
            paths=[input_path],
            label="cif_fob_required_input",
        ),
        "reference_inputs": source_files_fingerprint(
            paths=reference_paths,
            label="cif_fob_reference_inputs",
        ),
        "implementation": implementation_fp,
    }

    output_path = cfg.product_path(
        "corrected_trade"
    )

    audit_path = output_path.with_suffix(
        ".audit.csv"
    )

    return StepPlan(
        step_name=step_name,
        config_path=config_path,
        config_fingerprint=_step_config_fingerprint(
            config_path=config_path,
            resolved_config=cfg,
            pipeline_cfg=pipeline_cfg,
            include_scope=True,
        ),
        input_fingerprint=input_fp,
        output_path=output_path,
        audit_path=audit_path,
        runner=lambda run_id, force: _run_cif_fob_step(
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
            run_id=run_id,
            force=force,
        ),
    )


def _build_trade_sync_plan(
    *,
    step_name: str,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
) -> StepPlan:
    cfg = TradeSyncConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    input_path = cfg.input_path()

    trade_sync_source_dir = Path(
        run_trade_sync_workflow.__code__.co_filename
    ).parent

    implementation_fp = source_files_fingerprint(
        paths=list(
            trade_sync_source_dir.rglob("*.py")
        ),
        label="trade_sync_implementation",
    )

    input_fp = {
        "trade_sync_input": file_list_fingerprint(
            paths=[input_path],
            label="trade_sync_required_input",
        ),
        "implementation": implementation_fp,
    }

    output_path = cfg.product_path(
        "reconciled_trade"
    )
    audit_path = cfg.product_path(
        "reconciliation_audit"
    )

    return StepPlan(
        step_name=step_name,
        config_path=config_path,
        config_fingerprint=_step_config_fingerprint(
            config_path=config_path,
            resolved_config=cfg,
            pipeline_cfg=pipeline_cfg,
            include_scope=True,
        ),
        input_fingerprint=input_fp,
        output_path=output_path,
        audit_path=audit_path,
        runner=lambda run_id, force: _run_trade_sync_step(
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
            run_id=run_id,
            force=force
        ),
    )


def _build_value_plan(
    *,
    step_name: str,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
) -> StepPlan:
    cfg = ValueConfig.from_yaml(
        config_path,
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    input_path = cfg.input_path()
    deflator_source_path = cfg.deflator_source_path()

    value_source_dir = Path(
        run_value_workflow.__code__.co_filename
    ).parent

    implementation_fp = source_files_fingerprint(
        paths=list(
            value_source_dir.rglob("*.py")
        ),
        label="value_implementation",
    )

    input_fp = {
        "value_input": file_list_fingerprint(
            paths=[input_path],
            label="value_required_input",
        ),
        "reference_inputs": source_files_fingerprint(
            paths=[deflator_source_path],
            label="value_reference_inputs",
        ),
        "implementation": implementation_fp,
    }

    output_path = cfg.product_path(
        "deflated_trade"
    )

    audit_path = cfg.product_path(
        "audit"
    )

    return StepPlan(
        step_name=step_name,
        config_path=config_path,
        config_fingerprint=_step_config_fingerprint(
            config_path=config_path,
            resolved_config=cfg,
            pipeline_cfg=pipeline_cfg,
            include_scope=True,
        ),
        input_fingerprint=input_fp,
        output_path=output_path,
        audit_path=audit_path,
        runner=lambda run_id, force: _run_value_step(
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
            run_id=run_id,
            force=force,
        ),
    )

def _step_config_fingerprint(
    *,
    config_path: Path,
    resolved_config: Any,
    pipeline_cfg: PipelineConfig,
    include_scope: bool,
) -> dict[str, Any]:
    """
    Build raw and semantic fingerprints for one pipeline step.

    The raw YAML hash is retained for provenance.

    Checkpoint reuse is controlled by:
      - the semantic hash of the resolved step configuration;
      - the separately recorded pipeline scope, when applicable.

    Execution-only thread count is excluded because it does not change
    the produced scientific result. Scope is excluded from the semantic
    hash when it is recorded separately.
    """
    raw = config_fingerprint(
        config_path
    )

    excluded_paths: list[tuple[str, ...]] = [
        ("engine", "threads"),
    ]

    if include_scope:
        excluded_paths.append(
            ("scope",)
        )

    semantic = semantic_config_fingerprint(
        config=resolved_config,
        excluded_paths=tuple(
            excluded_paths
        ),
    )

    out: dict[str, Any] = {
        "step_config": {
            "path": raw["path"],
            "raw_sha256": raw["sha256"],
            "semantic_sha256": semantic["sha256"],
            "semantic_excluded_paths": semantic[
                "excluded_paths"
            ],
        },
    }

    if include_scope:
        out["pipeline_scope"] = {
            "target_hs": pipeline_cfg.scope.target_hs,
            "min_year": pipeline_cfg.scope.min_year,
            "max_year": pipeline_cfg.scope.max_year,
        }

    return out


def _step_config_fingerprint_dep(
    *,
    config_path: Path,
    resolved_config: Any,
    pipeline_cfg: PipelineConfig,
    include_scope: bool,
) -> dict[str, Any]:
    """
    Build raw and semantic fingerprints for one pipeline step.

    The raw file hash is retained for provenance. Reuse validation uses
    the semantic hash, which excludes execution-only thread count.
    """
    raw = config_fingerprint(
        config_path
    )

    semantic = semantic_config_fingerprint(
        config=resolved_config,
        excluded_paths=(
            ("engine", "threads"),
        ),
    )

    out: dict[str, Any] = {
        "step_config": {
            "path": raw["path"],
            "raw_sha256": raw["sha256"],
            "semantic_sha256": semantic["sha256"],
            "semantic_excluded_paths": semantic[
                "excluded_paths"
            ],
        },
    }

    if include_scope:
        out["pipeline_scope"] = {
            "target_hs": pipeline_cfg.scope.target_hs,
            "min_year": pipeline_cfg.scope.min_year,
            "max_year": pipeline_cfg.scope.max_year,
        }

    return out

def _step_config_fingerprint_deprecated(
    *,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
    include_scope: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "step_config": config_fingerprint(config_path),
    }

    if include_scope:
        out["pipeline_scope"] = {
            "target_hs": pipeline_cfg.scope.target_hs,
            "min_year": pipeline_cfg.scope.min_year,
            "max_year": pipeline_cfg.scope.max_year,
        }

    return out

def _build_step_plan(
    *,
    step_name: str,
    config_path: Path,
    pipeline_cfg: PipelineConfig,
) -> StepPlan:

    if step_name == "staging":
        return _build_staging_plan(
            step_name=step_name,
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
        )

    if step_name == "hs_sync":
        return _build_hs_sync_plan(
            step_name=step_name,
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
        )

    if step_name == "quantity":
        return _build_quantity_plan(
            step_name=step_name,
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
        )

    if step_name == "cif_fob":
        return _build_cif_fob_plan(
            step_name=step_name,
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
        )
    
    if step_name == "trade_sync":
        return _build_trade_sync_plan(
            step_name=step_name,
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
        )


    if step_name == "value":
        return _build_value_plan(
            step_name=step_name,
            config_path=config_path,
            pipeline_cfg=pipeline_cfg,
        )

    raise NotImplementedError(
        f"No pipeline plan implemented for step {step_name!r}."
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


def _result_audit_path(
    result: dict[str, Any],
    fallback: Path,
) -> Path:
    value = (
        result.get("reconciliation_audit_path")
        or result.get("audit_path")
        or result.get("audit_file")
        or result.get("final_audit")
        or fallback
    )

    return Path(value)


def run_pipeline(
    *,
    force_run_all: bool | None = None,
    require_confirmation: bool | None = None,
    target_hs: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    threads: int | None = None) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


    cfg = PipelineConfig.from_yaml()
    cfg = cfg.with_cli_overrides(
            force_run_all=force_run_all,
            require_confirmation=require_confirmation,
            target_hs=target_hs,
            min_year=min_year,
            max_year=max_year,
            threads=threads,
        )

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

    try:
        decisions = _build_step_decisions(
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
        )
    except Exception as exc:
        logger.exception("Could not build pipeline preflight plan.")
        raise SystemExit(1) from exc
    
    if cfg.pipeline.require_confirmation:
        confirmed = _confirm_pipeline_run(
            cfg=cfg,
            run_id=run_id,
            manifest_path=manifest_path,
            log_path=log_path,
            checkpoint_dir=checkpoint_dir,
            decisions=decisions,
        )

        if not confirmed:
            print(f"{get_symbols('flag')}Pipeline run cancelled by user.")
            return {
                "status": "cancelled",
                "run_id": run_id,
                "manifest_path": manifest_path,
                "log_path": log_path,
                "n_steps": len(decisions),
            }

    if cfg.pipeline.force_run_all:
        logger.warning("%s FORCE RUN ALL is enabled — valid checkpoints will be ignored.", get_symbols("warning_light"))


    if not cfg.pipeline.enabled:
        logger.warning("%s Pipeline is disabled in config. Nothing to run.", get_symbols("warning_light"))
        return {
            "status": "disabled",
            "run_id": run_id,
            "manifest_path": manifest_path,
            "log_path": log_path,
        }

    manifest_rows: list[dict[str, Any]] = []
    pipeline_status = "completed"


    with timed_pipeline_workflow(logger=logger, workflow_name="VACODA master pipeline"):

        logger.info("Run ID: %s", run_id)
        logger.info("Pipeline: %s", cfg.pipeline.name)
        logger.info("Engine name: %s", cfg.engine.name)
        logger.info("Engine threads: %s", cfg.engine.threads)
        logger.info("Enabled steps: %s", ", ".join(cfg.pipeline.enabled_steps))
        logger.info("Checkpoint reuse: %s", cfg.checkpoint.reuse_successful_steps)
        logger.info("Force run all: %s", cfg.pipeline.force_run_all)
        logger.info("Checkpoint dir: %s", checkpoint_dir)
        logger.info("Manifest: %s", manifest_path)
        logger.info("Log file: %s", log_path)

        
        #for step_name, step_cfg in cfg.enabled_steps():
        for decision in decisions:
            step_name = decision.step_name
            step_cfg = decision.step_config
            plan = decision.plan
            step_checkpoint_path = decision.checkpoint_path

            logger.info("")
            log_pipeline_step_banner(logger, step_name,)
            logger.info("Config: %s", step_cfg.config)
            logger.info("Expected status: %s", step_cfg.expected_status)
            logger.info("Required ready flag: %s", step_cfg.required_ready_flag)

            if decision.action == "skip":
                checkpoint = load_checkpoint(step_checkpoint_path)

                logger.info("SKIP — %s", step_name)
                logger.info("  Reason: %s", decision.reason)
                logger.info(" %s  Checkpoint: %s", get_symbols("checkpoint"), step_checkpoint_path)
                logger.info(
                    " %s  Output: %s",
                    get_symbols("output"), checkpoint.get("output", {}).get("path") if checkpoint else None,
                )
                logger.info(
                    " %s  Audit: %s",
                    get_symbols("audit"), checkpoint.get("audit", {}).get("path") if checkpoint else None,
                )

                manifest_rows.append(
                    _manifest_row(
                        run_id=run_id,
                        step_name=step_name,
                        status="skipped_checkpoint",
                        ready_flag=step_cfg.required_ready_flag,
                        ready_value=checkpoint.get("ready_value") if checkpoint else None,
                        duration_seconds=0.0,
                        output_path=checkpoint.get("output", {}).get("path") if checkpoint else None,
                        audit_path=checkpoint.get("audit", {}).get("path") if checkpoint else None,
                        n_failed_audit_rows=(
                            checkpoint.get("audit", {}).get("n_failed_audit_rows")
                            if checkpoint
                            else None
                        ),
                        message="Reused valid previous checkpoint.",
                    )
                )

                continue

            start = perf_counter()

            try:
                force_internal_rebuild = (cfg.pipeline.force_run_all)

                result = plan.runner(
                    run_id,
                    force_internal_rebuild,
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

                output_path = _result_output_path(result, plan.output_path)
                audit_path = _result_audit_path(result, plan.audit_path)

                if ok:
                    logger.info("DONE — %s (%.2fs)", step_name, duration_seconds)
                    logger.info("  Status: %s", status)
                    logger.info("  Ready flag %s: %s", step_cfg.required_ready_flag, ready_value)

                    if cfg.checkpoint.enabled:

                        checkpoint_plan = _build_step_plan(
                                            step_name=step_name,
                                            config_path=step_cfg.config,
                                            pipeline_cfg=cfg,
                                        )

                        write_checkpoint(
                            path=step_checkpoint_path,
                            run_id=run_id,
                            step_name=step_name,
                            status=status,
                            ready_flag=step_cfg.required_ready_flag,
                            ready_value=ready_value,
                            config=checkpoint_plan.config_fingerprint,
                            input_fingerprint=checkpoint_plan.input_fingerprint,
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

        logger.info("")
        logger.info("=" * 72)
        logger.info("%s Status: %s", get_symbols("right_finger"), pipeline_status)
        logger.info(" Manifest: %s",manifest_path)
        logger.info(" Log file: %s", log_path)
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
