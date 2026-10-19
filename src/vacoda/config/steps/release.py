"""VACODA configuration reader for configs/steps/release.yaml.

Scope and engine come from the master pipeline. The workflow can pass
actual checkpoint artifact paths; standalone callers can resolve them
using the configured prepared filename and a training-definition name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vacoda.config.base import (
    get_step_config_path,
    load_yaml_config,
    require_bool,
    require_dict,
    require_str,
)
from vacoda.config.paths import get_data_dir, get_output_dir
from vacoda.config.pipeline import PipelineEngineConfig, PipelineScopeConfig

RELEASE_CONFIG_PATH = get_step_config_path("release")


def load_release_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_yaml_config(config_path or RELEASE_CONFIG_PATH)


def _render_value(value: Any, *, target_hs: str) -> Any:
    if isinstance(value, dict):
        return {k: _render_value(v, target_hs=target_hs) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_value(v, target_hs=target_hs) for v in value]
    if isinstance(value, str):
        return value.format(
            target_hs=target_hs.upper(),
            target_hs_upper=target_hs.upper(),
            target_hs_lower=target_hs.lower(),
        )
    return value


def _str_list(value: Any, *, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a nonempty list.")
    fields = tuple(require_str(x, key=key).strip() for x in value)
    if any(not field for field in fields):
        raise ValueError(f"{key} must not contain blank columns.")
    if len(fields) != len(set(fields)):
        raise ValueError(f"{key} must not contain duplicate columns.")
    return fields


@dataclass(frozen=True)
class ReleaseInputFileConfig:
    filename: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "ReleaseInputFileConfig":
        filename = require_str(cfg.get("filename"), key=f"{key}.filename")
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".parquet":
            raise ValueError(f"{key}.filename must be a relative Parquet path.")
        return cls(filename=filename)


@dataclass(frozen=True)
class ReleaseInputConfig:
    stage: str
    subdir: str
    format: str
    prepared: ReleaseInputFileConfig
    inference: ReleaseInputFileConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ReleaseInputConfig":
        stage = require_str(cfg.get("stage"), key="release.input.stage").strip().lower()
        fmt = require_str(cfg.get("format"), key="release.input.format").strip().lower()
        subdir = require_str(cfg.get("subdir"), key="release.input.subdir", allow_empty=True)
        if stage != "processed":
            raise ValueError("release.input.stage must be 'processed'.")
        if fmt != "parquet":
            raise ValueError("release.input.format must be 'parquet'.")
        if Path(subdir).is_absolute() or ".." in Path(subdir).parts:
            raise ValueError("release.input.subdir must be a safe relative path.")
        return cls(
            stage=stage,
            subdir=subdir,
            format=fmt,
            prepared=ReleaseInputFileConfig.from_dict(
                require_dict(cfg.get("prepared"), key="release.input.prepared"),
                key="release.input.prepared",
            ),
            inference=ReleaseInputFileConfig.from_dict(
                require_dict(cfg.get("inference"), key="release.input.inference"),
                key="release.input.inference",
            ),
        )

    def resolve_dir(self) -> Path:
        return get_data_dir(self.stage) / self.subdir

    def prepared_path(self) -> Path:
        return self.resolve_dir() / self.prepared.filename

    def inference_path(self, *, training_definition: str) -> Path:
        # Exactly the model/<training_definition>/inference structure used
        # by the Validation inference step. Never guess the definition.
        if (
            not training_definition
            or training_definition in {".", ".."}
            or Path(training_definition).name != training_definition
            or "/" in training_definition
            or "\\" in training_definition
        ):
            raise ValueError("training_definition must be one directory name.")
        return (
            self.resolve_dir()
            / "model"
            / training_definition
            / "inference"
            / self.inference.filename
        )


@dataclass(frozen=True)
class ReleaseOutputConfig:
    stage: str
    subdir: str
    format: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ReleaseOutputConfig":
        stage = require_str(cfg.get("stage"), key="release.output.stage").lower()
        fmt = require_str(cfg.get("format"), key="release.output.format").lower()
        if fmt != "parquet":
            raise ValueError("release.output.format must be 'parquet'.")
        if stage != "release":
            raise ValueError("release.output.stage must be 'release'.")
        subdir = require_str(
            cfg.get("subdir"), key="release.output.subdir", allow_empty=True
        )
        if Path(subdir).is_absolute() or ".." in Path(subdir).parts:
            raise ValueError("release.output.subdir must be a safe relative path.")
        return cls(stage=stage, subdir=subdir, format=fmt)

    def resolve_dir(self) -> Path:
        return get_data_dir(self.stage) / self.subdir


@dataclass(frozen=True)
class ReleaseProductConfig:
    enable: bool
    filename: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "ReleaseProductConfig":
        filename = require_str(cfg.get("filename"), key=f"{key}.filename")
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(f"{key}.filename must be a safe relative path.")
        if Path(filename).suffix.lower() != ".parquet":
            raise ValueError(f"{key}.filename must end in .parquet.")
        return cls(
            enable=require_bool(cfg.get("enable"), key=f"{key}.enable"),
            filename=filename,
        )


@dataclass(frozen=True)
class ReleaseProductsConfig:
    prepare_input: ReleaseProductConfig
    release: ReleaseProductConfig
    release_audit: ReleaseProductConfig
    diagnostics: ReleaseProductConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ReleaseProductsConfig":
        def product(name: str) -> ReleaseProductConfig:
            key = f"release.products.{name}"
            return ReleaseProductConfig.from_dict(
                require_dict(cfg.get(name), key=key), key=key
            )
        result = cls(
            prepare_input=product("prepare_input"),
            release=product("release"),
            release_audit=product("release_audit"),
            diagnostics=product("diagnostics"),
        )
        if not result.release.enable or not result.release_audit.enable:
            raise ValueError("Release and release_audit are mandatory products.")
        return result


@dataclass(frozen=True)
class ReleaseSchemaConfig:
    standard: tuple[str, ...]
    enriched_additional: tuple[str, ...]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ReleaseSchemaConfig":
        standard = _str_list(cfg.get("standard"), key="release.schema.standard")
        additional = _str_list(
            cfg.get("enriched_additional"), key="release.schema.enriched_additional"
        )
        overlap = set(standard) & set(additional)
        if overlap:
            raise ValueError("Release schemas overlap: " + ", ".join(sorted(overlap)))
        required = {
            "year", "exporter_iso", "importer_iso", "target_cmd",
            "trade_value_reconciled", "trade_synced_quantity_kg",
            "model_prediction_quantity_kg", "model_prediction_available",
            "manual_review_required",
        }
        if not required.issubset(standard):
            raise ValueError("Release standard schema missing required fields: "
                             + ", ".join(sorted(required - set(standard))))
        return cls(standard=standard, enriched_additional=additional)

    @property
    def enriched(self) -> tuple[str, ...]:
        return self.standard + self.enriched_additional


@dataclass(frozen=True)
class ReleaseLoggingConfig:
    level: str
    subdir: str
    file: str
    timestamp: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ReleaseLoggingConfig":
        return cls(
            level=require_str(cfg.get("level"), key="release.logging.level").upper(),
            subdir=require_str(
                cfg.get("subdir"), key="release.logging.subdir", allow_empty=True
            ),
            file=require_str(cfg.get("file"), key="release.logging.file"),
            timestamp=require_bool(cfg.get("timestamp"), key="release.logging.timestamp"),
        )

    def resolve_log_dir(self) -> Path:
        parts = list(Path(self.subdir).parts)
        if parts and parts[0].lower() == "logs":
            parts = parts[1:]
        return get_output_dir("logs") / Path(*parts) if parts else get_output_dir("logs")

    def resolve_log_file(self) -> Path:
        return self.resolve_log_dir() / self.file


@dataclass(frozen=True)
class ReleaseArtifactsConfig:
    enable: bool
    kind: str
    subdir: str
    filename: str
    timestamp: bool
    write_latest: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ReleaseArtifactsConfig":
        return cls(
            enable=require_bool(cfg.get("enable"), key="release.artifacts.enable"),
            kind=require_str(cfg.get("kind"), key="release.artifacts.kind").lower(),
            subdir=require_str(cfg.get("subdir"), key="release.artifacts.subdir", allow_empty=True),
            filename=require_str(cfg.get("filename"), key="release.artifacts.filename"),
            timestamp=require_bool(cfg.get("timestamp"), key="release.artifacts.timestamp"),
            write_latest=require_bool(cfg.get("write_latest"), key="release.artifacts.write_latest"),
        )

    def resolve_dir(self) -> Path:
        return get_output_dir(self.kind) / self.subdir

    def timestamped_path(self, run_id: str) -> Path:
        suffix = f"_{run_id}" if self.timestamp else ""
        return self.resolve_dir() / f"{self.filename}{suffix}.md"

    def latest_path(self) -> Path:
        return self.resolve_dir() / f"{self.filename}_latest.md"


@dataclass(frozen=True)
class ReleaseConfig:
    input: ReleaseInputConfig
    output: ReleaseOutputConfig
    products: ReleaseProductsConfig
    schema: ReleaseSchemaConfig
    logging: ReleaseLoggingConfig
    artifacts: ReleaseArtifactsConfig
    scope: PipelineScopeConfig
    engine: PipelineEngineConfig

    @classmethod
    def from_yaml(
        cls, config_path: Path | None = None, *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "ReleaseConfig":
        return cls.from_dict(
            load_release_config(config_path), scope=scope, engine=engine
        )

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "ReleaseConfig":
        if scope is None or engine is None:
            raise ValueError("ReleaseConfig requires pipeline scope and engine.")
        root = require_dict(raw.get("release"), key="release")
        rendered = require_dict(
            _render_value(root, target_hs=scope.target_hs), key="release"
        )
        return cls(
            input=ReleaseInputConfig.from_dict(
                require_dict(rendered.get("input"), key="release.input")
            ),
            output=ReleaseOutputConfig.from_dict(
                require_dict(rendered.get("output"), key="release.output")
            ),
            products=ReleaseProductsConfig.from_dict(
                require_dict(rendered.get("products"), key="release.products")
            ),
            schema=ReleaseSchemaConfig.from_dict(
                require_dict(rendered.get("schema"), key="release.schema")
            ),
            logging=ReleaseLoggingConfig.from_dict(
                require_dict(rendered.get("logging"), key="release.logging")
            ),
            artifacts=ReleaseArtifactsConfig.from_dict(
                require_dict(rendered.get("artifacts"), key="release.artifacts")
            ),
            scope=scope,
            engine=engine,
        )

    def input_dir(self) -> Path:
        return self.input.resolve_dir()

    def prepared_input_path(self) -> Path:
        return self.input.prepared_path()

    def inference_input_path(self, *, training_definition: str) -> Path:
        return self.input.inference_path(training_definition=training_definition)

    def output_dir(self) -> Path:
        return self.output.resolve_dir()

    def product_path(self, product_name: str, *, release_id: str | None = None) -> Path:
        products = {
            "prepare_input": self.products.prepare_input,
            "release": self.products.release,
            "release_audit": self.products.release_audit,
            "diagnostics": self.products.diagnostics,
        }
        if product_name not in products:
            raise KeyError(f"Unknown Release product: {product_name!r}")
        base = self.output_dir()
        if release_id is not None:
            if not release_id or Path(release_id).name != release_id or release_id in {".", ".."} or "\\" in release_id:
                raise ValueError("release_id must be one safe directory name.")
            base = base / release_id
        return base / products[product_name].filename

    def enriched_path(self, *, release_id: str | None = None) -> Path:
        base = self.product_path("release", release_id=release_id)
        return base.with_name(f"{base.stem}_enriched{base.suffix}")

    def metadata_path(self, *, enriched: bool = False,
                      release_id: str | None = None) -> Path:
        parquet = (self.enriched_path(release_id=release_id) if enriched
                   else self.product_path("release", release_id=release_id))
        return parquet.with_suffix(".metadata.json")

    def enabled_products(self) -> dict[str, bool]:
        return {
            "prepare_input": self.products.prepare_input.enable,
            "release": self.products.release.enable,
            "release_audit": self.products.release_audit.enable,
            "diagnostics": self.products.diagnostics.enable,
        }

    def log_path(self) -> Path:
        return self.logging.resolve_log_file()

    def artifacts_path(self, run_id: str) -> Path:
        return self.artifacts.timestamped_path(run_id)

    def artifacts_latest_path(self) -> Path:
        return self.artifacts.latest_path()
