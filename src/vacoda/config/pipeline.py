"""
vacoda.config.pipeline

Defines the configuration schema for the VACODA master pipeline, including pipeline metadata, step configurations, 
manifest settings, and logging preferences. 

This module provides classes to load and validate the pipeline configuration from a YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vacoda.config.base import CONFIG_DIR, load_yaml_config
from vacoda.config.paths import get_output_dir


PIPELINE_CONFIG_PATH = CONFIG_DIR / "pipeline.yaml"


def require_dict(value: Any, *, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a dictionary.")
    return value


def require_str(value: Any, *, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{key} must be a non-empty string.")
    return value


@dataclass(frozen=True)
class PipelineInfoConfig:
    name: str
    version: int
    enabled: bool
    enabled_steps: list[str]
    fail_fast: bool
    allow_skip_existing: bool
    force_run_all: bool
    require_confirmation: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PipelineInfoConfig":
        enabled_steps = cfg.get("enabled_steps", [])
        if not isinstance(enabled_steps, list):
            raise TypeError("pipeline.enabled_steps must be a list.")

        return cls(
            name=require_str(cfg.get("name"), key="pipeline.name"),
            version=int(cfg.get("version", 1)),
            enabled=bool(cfg.get("enabled", True)),
            enabled_steps=[str(step) for step in enabled_steps],
            fail_fast=bool(cfg.get("fail_fast", True)),
            allow_skip_existing=bool(cfg.get("allow_skip_existing", True)),
            force_run_all=bool(cfg.get("force_run_all", False)),
            require_confirmation=bool(cfg.get("require_confirmation", True)),
        )


@dataclass(frozen=True)
class PipelineStepConfig:
    enabled: bool
    config: Path
    expected_status: list[str]
    required_ready_flag: str | None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "PipelineStepConfig":
        expected_status = cfg.get("expected_status", ["completed"])
        if not isinstance(expected_status, list):
            raise TypeError(f"{key}.expected_status must be a list.")

        required_ready_flag = cfg.get("required_ready_flag")
        if required_ready_flag is not None:
            required_ready_flag = require_str(
                required_ready_flag,
                key=f"{key}.required_ready_flag",
            )

        return cls(
            enabled=bool(cfg.get("enabled", True)),
            config=Path(require_str(cfg.get("config"), key=f"{key}.config")),
            expected_status=[str(status) for status in expected_status],
            required_ready_flag=required_ready_flag,
        )


@dataclass(frozen=True)
class PipelineScopeConfig:
    target_hs: str
    min_year: int
    max_year: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PipelineScopeConfig":
        return cls(
            target_hs=require_str(cfg.get("target_hs", "H5"), key="scope.target_hs").strip().upper(),
            min_year=int(cfg.get("min_year", 1962)),
            max_year=int(cfg.get("max_year", 9999)),
        )


@dataclass(frozen=True)
class PipelineEngineConfig:
    name: str
    threads: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PipelineEngineConfig":
        return cls(
            name=require_str(cfg.get("name", "duckdb"), key="engine.name").strip().lower(),
            threads=int(cfg.get("threads", 16)),
        )


@dataclass(frozen=True)
class PipelineManifestConfig:
    kind: str
    subdir: str
    filename: str
    timestamp: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PipelineManifestConfig":
        return cls(
            kind=require_str(cfg.get("kind", "manifests"), key="manifest.kind"),
            subdir=require_str(cfg.get("subdir", "pipeline"), key="manifest.subdir"),
            filename=require_str(cfg.get("filename"), key="manifest.filename"),
            timestamp=bool(cfg.get("timestamp", True)),
        )

    def resolve_path(self) -> Path:
        return get_output_dir(self.kind) / self.subdir / self.filename


@dataclass(frozen=True)
class PipelineCheckpointConfig:
    enabled: bool
    kind: str
    subdir: str
    reuse_successful_steps: bool
    fingerprint_mode: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PipelineCheckpointConfig":
        fingerprint_mode = require_str(
            cfg.get("fingerprint_mode", "strong_metadata"),
            key="checkpoint.fingerprint_mode",
        )

        allowed_modes = {"fast", "strong_metadata", "content_hash"}
        if fingerprint_mode not in allowed_modes:
            raise ValueError(
                "checkpoint.fingerprint_mode must be one of "
                f"{sorted(allowed_modes)!r}, got {fingerprint_mode!r}."
            )

        return cls(
            enabled=bool(cfg.get("enabled", True)),
            kind=require_str(cfg.get("kind", "manifests"), key="checkpoint.kind"),
            subdir=require_str(cfg.get("subdir", "pipeline/checkpoints"), key="checkpoint.subdir"),
            reuse_successful_steps=bool(cfg.get("reuse_successful_steps", True)),
            fingerprint_mode=fingerprint_mode,
        )

    def resolve_dir(self) -> Path:
        return get_output_dir(self.kind) / self.subdir


@dataclass(frozen=True)
class PipelineLoggingConfig:
    level: str
    subdir: str
    file: str
    timestamp: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PipelineLoggingConfig":
        return cls(
            level=require_str(cfg.get("level", "INFO"), key="logging.level"),
            subdir=require_str(cfg.get("subdir", "pipeline"), key="logging.subdir"),
            file=require_str(cfg.get("file"), key="logging.file"),
            timestamp=bool(cfg.get("timestamp", True)),
        )

    def resolve_log_file(self) -> Path:
        return get_output_dir("logs") / self.subdir / self.file


@dataclass(frozen=True)
class PipelineConfig:
    pipeline: PipelineInfoConfig
    scope: PipelineScopeConfig
    engine: PipelineEngineConfig
    steps: dict[str, PipelineStepConfig]
    checkpoint: PipelineCheckpointConfig
    manifest: PipelineManifestConfig
    logging: PipelineLoggingConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        steps_raw = require_dict(raw.get("steps"), key="steps")

        return cls(
            pipeline=PipelineInfoConfig.from_dict(
                require_dict(raw.get("pipeline"), key="pipeline")
            ),
            scope=PipelineScopeConfig.from_dict(
                require_dict(raw.get("scope", {}), key="scope")
            ),
            engine=PipelineEngineConfig.from_dict(
                require_dict(raw.get("engine", {}), key="engine")
            ),
            steps={
                name: PipelineStepConfig.from_dict(
                    require_dict(step_cfg, key=f"steps.{name}"),
                    key=f"steps.{name}",
                )
                for name, step_cfg in steps_raw.items()
            },
            checkpoint=PipelineCheckpointConfig.from_dict(
                require_dict(raw.get("checkpoint"), key="checkpoint")
            ),
            manifest=PipelineManifestConfig.from_dict(
                require_dict(raw.get("manifest"), key="manifest")
            ),
            logging=PipelineLoggingConfig.from_dict(
                require_dict(raw.get("logging"), key="logging")
            ),
        )

    @classmethod
    def from_yaml(cls) -> "PipelineConfig":
        raw = load_yaml_config(PIPELINE_CONFIG_PATH)

        if raw is None:
            raise ValueError(f"Empty pipeline config: {PIPELINE_CONFIG_PATH}")

        return cls.from_dict(raw)

    def enabled_steps(self) -> list[tuple[str, PipelineStepConfig]]:
        out = []

        for step_name in self.pipeline.enabled_steps:
            if step_name not in self.steps:
                raise KeyError(f"Pipeline step {step_name!r} is enabled but not defined.")

            step = self.steps[step_name]
            if step.enabled:
                out.append((step_name, step))

        return out