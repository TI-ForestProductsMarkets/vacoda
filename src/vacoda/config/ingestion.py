# src/vacoda/config/ingestion.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vacoda.config.base import (
    get_step_config_path,
    load_yaml_config,
    require_bool,
    require_dict,
    require_int,
    require_str,
)
from vacoda.config.paths import get_data_dir, get_output_dir


INGESTION_CONFIG_PATH = get_step_config_path("ingestion")


def load_ingestion_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_yaml_config(config_path or INGESTION_CONFIG_PATH)


@dataclass(frozen=True)
class StagePathConfig:
    stage: str
    subdir: str = ""
    pattern: str | None = None
    file: str | None = None
    format: str | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "StagePathConfig":
        fmt = cfg.get("format")

        pattern = cfg.get("pattern")
        file = cfg.get("file")

        return cls(
            stage=require_str(cfg.get("stage"), key=f"{key}.stage").strip().lower(),
            subdir=require_str(
                cfg.get("subdir", ""),
                key=f"{key}.subdir",
                allow_empty=True,
            ),
            pattern=None if pattern is None else require_str(pattern, key=f"{key}.pattern"),
            file=None if file is None else require_str(file, key=f"{key}.file"),
            format=None if fmt is None else require_str(fmt, key=f"{key}.format").strip().lower(),
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_input_glob(self) -> Path:
        if self.pattern is None:
            raise ValueError("input.pattern is required for multi-file input.")
        return self.resolve_dir() / self.pattern

    def resolve_file(self) -> Path:
        if self.file is None:
            raise ValueError("file is required for this path config.")
        return self.resolve_dir() / self.file
        
@dataclass(frozen=True)
class IngestionInfoConfig:
    name: str
    mode: str
    source: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "IngestionInfoConfig":
        return cls(
            name=require_str(cfg.get("name"), key="ingestion.name"),
            mode=require_str(cfg.get("mode"), key="ingestion.mode").strip().lower(),
            source=require_str(cfg.get("source"), key="ingestion.source").strip().lower(),
        )


@dataclass(frozen=True)
class EngineConfig:
    name: str
    threads: int

    @classmethod
    def from_pipeline_engine(cls, engine: Any) -> "EngineConfig":
        return cls(
            name=require_str(engine.name, key="pipeline.engine.name").strip().lower(),
            threads=require_int(engine.threads, key="pipeline.engine.threads", min_value=1),
        )

@dataclass(frozen=True)
class RawSchemaConfig:
    required_fields: list[str]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "RawSchemaConfig":
        fields = cfg.get("required_fields", [])
        if not isinstance(fields, list):
            raise TypeError("schema.raw.required_fields must be a list.")

        return cls(
            required_fields=[
                require_str(item, key="schema.raw.required_fields")
                for item in fields
            ]
        )


@dataclass(frozen=True)
class StandardizedSchemaConfig:
    name: str
    drop_unmapped_fields: bool
    field_mapping: dict[str, str]
    required_fields: list[str]
    dtypes: dict[str, str]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "StandardizedSchemaConfig":
        mapping = require_dict(cfg.get("field_mapping", {}), key="schema.standardized.field_mapping")
        dtypes = require_dict(cfg.get("dtypes", {}), key="schema.standardized.dtypes")

        required_fields = cfg.get("required_fields", [])
        if not isinstance(required_fields, list):
            raise TypeError("schema.standardized.required_fields must be a list.")

        return cls(
            name=require_str(cfg.get("name"), key="schema.standardized.name"),
            drop_unmapped_fields=require_bool(
                cfg.get("drop_unmapped_fields", True),
                key="schema.standardized.drop_unmapped_fields",
            ),
            field_mapping={
                require_str(k, key="schema.standardized.field_mapping.<key>"):
                require_str(v, key=f"schema.standardized.field_mapping.{k}")
                for k, v in mapping.items()
            },
            required_fields=[
                require_str(item, key="schema.standardized.required_fields")
                for item in required_fields
            ],
            dtypes={
                require_str(k, key="schema.standardized.dtypes.<key>"):
                require_str(v, key=f"schema.standardized.dtypes.{k}")
                for k, v in dtypes.items()
            },
        )


@dataclass(frozen=True)
class SchemaConfig:
    name: str
    version: int
    strict: bool
    raw: RawSchemaConfig
    standardized: StandardizedSchemaConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SchemaConfig":
        return cls(
            name=require_str(cfg.get("name"), key="schema.name"),
            version=require_int(cfg.get("version", 1), key="schema.version", min_value=1),
            strict=require_bool(cfg.get("strict", False), key="schema.strict"),
            raw=RawSchemaConfig.from_dict(
                require_dict(cfg.get("raw", {}), key="schema.raw")
            ),
            standardized=StandardizedSchemaConfig.from_dict(
                require_dict(cfg.get("standardized", {}), key="schema.standardized")
            ),
        )


@dataclass(frozen=True)
class WriteConfig:
    overwrite: bool
    allow_empty: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "WriteConfig":
        return cls(
            overwrite=require_bool(cfg.get("overwrite", True), key="write.overwrite"),
            allow_empty=require_bool(cfg.get("allow_empty", False), key="write.allow_empty"),
        )


@dataclass(frozen=True)
class AuditConfig:
    enabled: bool
    kind: str
    subdir: str
    filename: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "AuditConfig":
        return cls(
            enabled=require_bool(cfg.get("enabled", True), key="audit.enabled"),
            kind=require_str(cfg.get("kind", "manifest"), key="audit.kind").strip().lower(),
            subdir=require_str(cfg.get("subdir", ""), key="audit.subdir", allow_empty=True),
            filename=require_str(cfg.get("filename"), key="audit.filename"),
        )

    def resolve_output_path(self) -> Path:
        base = get_output_dir("manifests" if self.kind in {"manifest", "manifests"} else self.kind)
        return base / self.subdir / self.filename


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    file: str
    timestamp: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "LoggingConfig":
        return cls(
            level=require_str(cfg.get("level", "INFO"), key="logging.level").upper(),
            file=require_str(cfg.get("file", "LOG_INGESTION_BUILD.log"), key="logging.file"),
            timestamp=require_bool(cfg.get("timestamp", True), key="logging.timestamp"),
        )

    def resolve_log_file(self) -> Path:

        return get_output_dir("logs") / self.file

@dataclass(frozen=True)
class IngestionConfig:
    ingestion: IngestionInfoConfig
    input: StagePathConfig
    output: StagePathConfig
    engine: EngineConfig
    schema: SchemaConfig
    write: WriteConfig
    audit: AuditConfig
    logging: LoggingConfig
    
    @classmethod
    def from_yaml(cls,
                config_path: Path | None = None,
                *,
                scope: Any | None = None,
                engine: Any | None = None) -> "IngestionConfig":
                return cls.from_dict(
                    load_ingestion_config(config_path),
                    scope=scope,
                    engine=engine,
                )

    @classmethod
    def from_dict(
                cls,
                raw: dict[str, Any],
                *,
                scope: Any | None = None,
                engine: Any | None = None) -> "IngestionConfig":
        if engine is None:
            raise ValueError("IngestionConfig requires pipeline engine.")

        return cls(
            ingestion=IngestionInfoConfig.from_dict(require_dict(raw.get("ingestion"), key="ingestion")),
            input=StagePathConfig.from_dict(require_dict(raw.get("input"), key="input"), key="input"),
            output=StagePathConfig.from_dict(require_dict(raw.get("output"), key="output"), key="output"),
            engine=(EngineConfig.from_pipeline_engine(engine)),
            schema=SchemaConfig.from_dict(require_dict(raw.get("schema"), key="schema")),
            write=WriteConfig.from_dict(require_dict(raw.get("write", {}), key="write")),
            audit=AuditConfig.from_dict(require_dict(raw.get("audit", {}), key="audit")),
            logging=LoggingConfig.from_dict(require_dict(raw.get("logging", {}), key="logging")),
        )

    def input_glob(self) -> Path:
        return self.input.resolve_input_glob()

    def output_path(self) -> Path:
        return self.output.resolve_file()

    def audit_path(self) -> Path:
        return self.audit.resolve_output_path()

    