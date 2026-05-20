"""
vacoda.config.dataset

Config reader for configs/dataset.yaml.

This module handles dataset-build configuration only.
It does not resolve global storage paths directly; paths are resolved via
vacoda.config.paths using stage/subdir/file information from this config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from vacoda.config.base import (
    CONFIG_DIR,
    get_required_block,
    get_required_value,
    load_yaml_config,
    require_bool,
    require_dict,
    require_int,
    require_str,
)


DATASET_CONFIG_NAME = "dataset.yaml"
DATASET_CONFIG_PATH = CONFIG_DIR / DATASET_CONFIG_NAME


@dataclass(frozen=True)
class DatasetInfoConfig:
    name: str
    mode: str
    source: str


@dataclass(frozen=True)
class DatasetInputConfig:
    stage: str
    subdir: str
    pattern: str = "*.csv"
    files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetOutputConfig:
    stage: str
    subdir: str
    format: str
    files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RawSchemaConfig:
    required_fields: list[str]


@dataclass(frozen=True)
class StandardizedSchemaConfig:
    name: str
    field_mapping: dict[str, str]
    drop_unmapped_fields: bool
    required_fields: list[str]
    dtypes: dict[str, str]


@dataclass(frozen=True)
class DatasetSchemaConfig:
    name: str
    strict: bool
    raw: RawSchemaConfig
    standardized: StandardizedSchemaConfig


@dataclass(frozen=True)
class DatasetWriteConfig:
    overwrite: bool = True
    allow_empty: bool = False


@dataclass(frozen=True)
class DatasetLoggingConfig:
    level: str = "INFO"
    file: str | None = "LOG_DATASET_BUILD.log"
    timestamp: bool = True


@dataclass(frozen=True)
class DatasetBuildConfig:
    dataset: DatasetInfoConfig
    input: DatasetInputConfig
    output: DatasetOutputConfig
    schema: DatasetSchemaConfig
    write: DatasetWriteConfig
    logging: DatasetLoggingConfig


def load_dataset_build_config(
    config_path: Path | None = None,
) -> DatasetBuildConfig:
    """Load configs/dataset.yaml as a typed DatasetBuildConfig."""
    cfg = load_dataset_config(config_path)

    return DatasetBuildConfig(
        dataset=DatasetInfoConfig(
            name=get_dataset_name(cfg),
            mode=get_dataset_mode(cfg),
            source=get_dataset_source(cfg),
        ),
        input=DatasetInputConfig(
            stage=get_dataset_input_stage(cfg),
            subdir=get_dataset_input_subdir(cfg),
            pattern=get_dataset_input_pattern(cfg),
            files=get_dataset_input_files(cfg),
        ),
        output=DatasetOutputConfig(
            stage=get_dataset_output_stage(cfg),
            subdir=get_dataset_output_subdir(cfg),
            format=get_dataset_output_format(cfg),
            files=get_dataset_output_files(cfg),
        ),
        schema=DatasetSchemaConfig(
            name=get_dataset_schema_name(cfg),
            strict=get_dataset_schema_strict_bool(cfg),
            raw=RawSchemaConfig(
                required_fields=get_dataset_raw_required_fields(cfg),
            ),
            standardized=StandardizedSchemaConfig(
                name=get_dataset_standardized_name(cfg),
                drop_unmapped_fields=get_dataset_standardized_schema_config(cfg).get("drop_unmapped_fields", False),
                field_mapping=get_dataset_field_mapping(cfg),
                required_fields=get_dataset_standardized_required_fields(cfg),
                dtypes=get_dataset_standardized_dtype_mapping(cfg),
            ),
        ),
        write=DatasetWriteConfig(
            overwrite=get_dataset_overwrite_bool(cfg),
            allow_empty=get_dataset_allow_empty_bool(cfg),
        ),
        logging=DatasetLoggingConfig(
            level=get_dataset_logging_level(cfg),
            file=get_dataset_logging_file(cfg),
            timestamp=get_dataset_logging_timestamp_bool(cfg),
        ),
    )


def load_dataset_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and validate configs/dataset.yaml."""
    cfg = load_yaml_config(config_path or DATASET_CONFIG_PATH)
    validate_dataset_config(cfg)
    return cfg


def validate_dataset_config(cfg: dict[str, Any]) -> None:
    """Validate required dataset.yaml structure."""
    dataset_cfg = get_required_block(
        cfg,
        "dataset",
        config_name=DATASET_CONFIG_NAME,
    )
    input_cfg = get_required_block(
        cfg,
        "input",
        config_name=DATASET_CONFIG_NAME,
    )
    output_cfg = get_required_block(
        cfg,
        "output",
        config_name=DATASET_CONFIG_NAME,
    )
    schema_cfg = get_required_block(
        cfg,
        "schema",
        config_name=DATASET_CONFIG_NAME,
    )

    # dataset block
    require_str(
        get_required_value(dataset_cfg, ("name",), config_name=f"{DATASET_CONFIG_NAME}: dataset"),
        key="dataset.name",
    )
    require_str(
        get_required_value(dataset_cfg, ("mode",), config_name=f"{DATASET_CONFIG_NAME}: dataset"),
        key="dataset.mode",
    )
    require_str(
        get_required_value(dataset_cfg, ("source",), config_name=f"{DATASET_CONFIG_NAME}: dataset"),
        key="dataset.source",
    )

    # input block
    require_str(
        get_required_value(input_cfg, ("stage",), config_name=f"{DATASET_CONFIG_NAME}: input"),
        key="input.stage",
    )
    require_str(
        get_required_value(input_cfg, ("subdir",), config_name=f"{DATASET_CONFIG_NAME}: input"),
        key="input.subdir",
        allow_empty=True,
    )

    # output block
    require_str(
        get_required_value(output_cfg, ("stage",), config_name=f"{DATASET_CONFIG_NAME}: output"),
        key="output.stage",
    )
    require_str(
        get_required_value(output_cfg, ("subdir",), config_name=f"{DATASET_CONFIG_NAME}: output"),
        key="output.subdir",
        allow_empty=True,
    )
    require_str(
        get_required_value(output_cfg, ("format",), config_name=f"{DATASET_CONFIG_NAME}: output"),
        key="output.format",
    )

    # schema block
    require_str(
        get_required_value(
            schema_cfg,
            ("name",),
            config_name=f"{DATASET_CONFIG_NAME}: schema",
        ),
        key="schema.name",
    )

    if "strict" in schema_cfg:
        require_bool(schema_cfg["strict"], key="schema.strict")

    raw_schema_cfg = get_required_block(
        schema_cfg,
        "raw",
        config_name=f"{DATASET_CONFIG_NAME}: schema",
    )

    standardized_schema_cfg = get_required_block(
        schema_cfg,
        "standardized",
        config_name=f"{DATASET_CONFIG_NAME}: schema",
    )

    standardized_required_fields = get_required_value(
        standardized_schema_cfg,
        ("required_fields",),
        config_name=f"{DATASET_CONFIG_NAME}: schema.standardized",
    )


    raw_required_fields = get_required_value(
                raw_schema_cfg,
                ("required_fields",),
                config_name=f"{DATASET_CONFIG_NAME}: schema.raw",
                )

    require_list_of_str(
        standardized_required_fields,
        key="schema.standardized.required_fields",
    )

    require_str(
        get_required_value(
            standardized_schema_cfg,
            ("name",),
            config_name=f"{DATASET_CONFIG_NAME}: schema.standardized",
        ),
        key="schema.standardized.name",
    )

    field_mapping = get_required_value(
        standardized_schema_cfg,
        ("field_mapping",),
        config_name=f"{DATASET_CONFIG_NAME}: schema.standardized",
    )

    require_dict_of_str(
        field_mapping,
        key="schema.standardized.field_mapping",
        )

    dtypes = get_required_value(
        standardized_schema_cfg,
        ("dtypes",),
        config_name=f"{DATASET_CONFIG_NAME}: schema.standardized",
    )

    require_dict_of_str(
        dtypes,
        key="schema.standardized.dtypes",
    )

    # optional blocks
    if "write" in cfg:
        write_cfg = get_required_block(
            cfg,
            "write",
            config_name=DATASET_CONFIG_NAME,
        )

        if "overwrite" in write_cfg:
            require_bool(write_cfg["overwrite"], key="write.overwrite")

        if "allow_empty" in write_cfg:
            require_bool(write_cfg["allow_empty"], key="write.allow_empty")

    if "logging" in cfg:
        logging_cfg = get_required_block(
            cfg,
            "logging",
            config_name=DATASET_CONFIG_NAME,
        )

        if "level" in logging_cfg:
            require_str(logging_cfg["level"], key="logging.level")

        if "file" in logging_cfg:
            require_str(logging_cfg["file"], key="logging.file")

        if "timestamp" in logging_cfg:
            require_bool(logging_cfg["timestamp"], key="logging.timestamp")


# ---------------------------------------------------------------------
# Small local validators
# You can move these to config.base later if you want them globally.
# ---------------------------------------------------------------------


def require_list(value: Any, *, key: str) -> list[Any]:
    """Require a config value to be a list."""
    if not isinstance(value, list):
        raise TypeError(
            f"'{key}' must be a list, got {value!r} "
            f"({type(value).__name__})."
        )

    return value


def require_list_of_str(value: Any, *, key: str) -> list[str]:
    """Require a config value to be a list of strings."""
    items = require_list(value, key=key)

    bad_items = [
        item
        for item in items
        if not isinstance(item, str)
    ]

    if bad_items:
        raise TypeError(
            f"'{key}' must contain only strings. Invalid item(s): {bad_items!r}"
        )

    return items


def require_dict_of_str(value: Any, *, key: str) -> dict[str, str]:
    """Require a config value to be a dict[str, str]."""
    items = require_dict(value, key=key)

    bad_keys = [
        item_key
        for item_key in items
        if not isinstance(item_key, str)
    ]
    bad_values = [
        item_value
        for item_value in items.values()
        if not isinstance(item_value, str)
    ]

    if bad_keys:
        raise TypeError(
            f"'{key}' must have only string keys. Invalid key(s): {bad_keys!r}"
        )

    if bad_values:
        raise TypeError(
            f"'{key}' must have only string values. Invalid value(s): {bad_values!r}"
        )

    return items


# ---------------------------------------------------------------------
# Top-level block getters
# ---------------------------------------------------------------------


def get_dataset_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return dataset config block."""
    if cfg is None:
        cfg = load_dataset_config()

    return get_required_block(
        cfg,
        "dataset",
        config_name=DATASET_CONFIG_NAME,
    )


def get_dataset_input_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return input config block."""
    if cfg is None:
        cfg = load_dataset_config()

    return get_required_block(
        cfg,
        "input",
        config_name=DATASET_CONFIG_NAME,
    )


def get_dataset_output_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return output config block."""
    if cfg is None:
        cfg = load_dataset_config()

    return get_required_block(
        cfg,
        "output",
        config_name=DATASET_CONFIG_NAME,
    )


def get_dataset_schema_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return schema config block."""
    if cfg is None:
        cfg = load_dataset_config()

    return get_required_block(
        cfg,
        "schema",
        config_name=DATASET_CONFIG_NAME,
    )


def get_dataset_write_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return write config block."""
    if cfg is None:
        cfg = load_dataset_config()

    return cfg.get(
        "write",
        {
            "overwrite": True,
            "allow_empty": False,
        },
    )


def get_dataset_logging_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return logging config block."""
    if cfg is None:
        cfg = load_dataset_config()

    return cfg.get(
        "logging",
        {
            "level": "INFO",
            "file": "LOG_DATASET_BUILD.log",
            "timestamp": True,
        },
    )


# ---------------------------------------------------------------------
# Dataset block getters
# ---------------------------------------------------------------------


def get_dataset_name(cfg: dict[str, Any] | None = None) -> str:
    """Return dataset name."""
    dataset_cfg = get_dataset_config(cfg)
    value = dataset_cfg.get("name")
    return require_str(value, key="dataset.name").strip()


def get_dataset_mode(cfg: dict[str, Any] | None = None) -> str:
    """Return dataset build mode."""
    dataset_cfg = get_dataset_config(cfg)
    value = dataset_cfg.get("mode", "per_file")
    mode = require_str(value, key="dataset.mode").strip().lower()

    valid_modes = {"per_file", "merged"}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid dataset.mode: {mode!r}. "
            f"Valid modes: {sorted(valid_modes)}"
        )

    return mode


def get_dataset_source(cfg: dict[str, Any] | None = None) -> str:
    """Return dataset source name."""
    dataset_cfg = get_dataset_config(cfg)
    value = dataset_cfg.get("source", "api")
    source = require_str(value, key="dataset.source").strip().lower()

    valid_sources = {"api", "local"}
    if source not in valid_sources:
        raise ValueError(
            f"Invalid dataset.source: {source!r}. "
            f"Valid sources: {sorted(valid_sources)}"
        )

    return source


# ---------------------------------------------------------------------
# Input block getters
# ---------------------------------------------------------------------


def get_dataset_input_stage(cfg: dict[str, Any] | None = None) -> str:
    """Return input data stage."""
    input_cfg = get_dataset_input_config(cfg)
    value = input_cfg.get("stage", "raw")
    return require_str(value, key="input.stage").strip()


def get_dataset_input_subdir(cfg: dict[str, Any] | None = None) -> str:
    """Return input subdirectory below the input stage."""
    input_cfg = get_dataset_input_config(cfg)
    value = input_cfg.get("subdir", "")
    return require_str(value, key="input.subdir", allow_empty=True).strip()


def get_dataset_input_pattern(cfg: dict[str, Any] | None = None) -> str:
    """Return input glob pattern."""
    input_cfg = get_dataset_input_config(cfg)
    value = input_cfg.get("pattern", "*.csv")
    return require_str(value, key="input.pattern").strip()


def get_dataset_input_files(cfg: dict[str, Any] | None = None) -> list[str]:
    """Return explicit input file names/patterns."""
    input_cfg = get_dataset_input_config(cfg)
    value = input_cfg.get("files", [])

    return require_list_of_str(value, key="input.files")


# ---------------------------------------------------------------------
# Output block getters
# ---------------------------------------------------------------------


def get_dataset_output_stage(cfg: dict[str, Any] | None = None) -> str:
    """Return output data stage."""
    output_cfg = get_dataset_output_config(cfg)
    value = output_cfg.get("stage", "processed")
    return require_str(value, key="output.stage").strip()


def get_dataset_output_subdir(cfg: dict[str, Any] | None = None) -> str:
    """Return output subdirectory below the output stage."""
    output_cfg = get_dataset_output_config(cfg)
    value = output_cfg.get("subdir", "")
    return require_str(value, key="output.subdir", allow_empty=True).strip()


def get_dataset_output_format(cfg: dict[str, Any] | None = None) -> str:
    """Return output file format."""
    output_cfg = get_dataset_output_config(cfg)
    value = output_cfg.get("format", "parquet")
    fmt = require_str(value, key="output.format").strip().lower()

    valid_formats = {"csv", "parquet"}
    if fmt not in valid_formats:
        raise ValueError(
            f"Invalid output.format: {fmt!r}. "
            f"Valid formats: {sorted(valid_formats)}"
        )

    return fmt


def get_dataset_output_files(cfg: dict[str, Any] | None = None) -> list[str]:
    """Return explicit output file names/patterns."""
    output_cfg = get_dataset_output_config(cfg)
    value = output_cfg.get("files", [])

    return require_list_of_str(value, key="output.files")


# ---------------------------------------------------------------------
# Schema block getters
# ---------------------------------------------------------------------


def get_dataset_schema_name(cfg: dict[str, Any] | None = None) -> str:
    """Return schema name."""
    schema_cfg = get_dataset_schema_config(cfg)
    value = schema_cfg.get("name")
    return require_str(value, key="schema.name").strip()


def get_dataset_schema_strict_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether schema validation should reject extra fields."""
    schema_cfg = get_dataset_schema_config(cfg)
    value = schema_cfg.get("strict", False)
    return require_bool(value, key="schema.strict")


def get_dataset_raw_schema_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return raw schema config block."""
    schema_cfg = get_dataset_schema_config(cfg)
    return get_required_block(
        schema_cfg,
        "raw",
        config_name=DATASET_CONFIG_NAME,
    )


def get_dataset_standardized_schema_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return standardized schema config block."""
    schema_cfg = get_dataset_schema_config(cfg)
    return get_required_block(
        schema_cfg,
        "standardized",
        config_name=DATASET_CONFIG_NAME,
    )


def get_dataset_raw_required_fields(
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Return required raw input fields."""
    raw_schema_cfg = get_dataset_raw_schema_config(cfg)
    value = raw_schema_cfg.get("required_fields", [])
    return require_list_of_str(value, key="schema.raw.required_fields")


def get_dataset_standardized_name(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return standardized schema name."""
    standardized_schema_cfg = get_dataset_standardized_schema_config(cfg)
    value = standardized_schema_cfg.get("name")
    return require_str(value, key="schema.standardized.name").strip()

def get_dataset_standardized_drop_unmapped_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether to drop unmapped fields in standardized output."""
    standardized_schema_cfg = get_dataset_standardized_schema_config(cfg)
    value = standardized_schema_cfg.get("drop_unmapped_fields", False)
    return require_bool(value, key="schema.standardized.drop_unmapped_fields")

def get_dataset_field_mapping(
    cfg: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return raw-to-standardized field mapping."""
    standardized_schema_cfg = get_dataset_standardized_schema_config(cfg)
    value = standardized_schema_cfg.get("field_mapping", {})
    return require_dict_of_str(value, key="schema.standardized.field_mapping")


def get_dataset_standardized_required_fields(
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Return required standardized fields."""
    standardized_schema_cfg = get_dataset_standardized_schema_config(cfg)
    value = standardized_schema_cfg.get("required_fields", [])
    return require_list_of_str(
        value,
        key="schema.standardized.required_fields",
    )


def get_dataset_standardized_dtype_mapping(
    cfg: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return standardized dtype mapping."""
    standardized_schema_cfg = get_dataset_standardized_schema_config(cfg)
    value = standardized_schema_cfg.get("dtypes", {})
    return require_dict_of_str(value, key="schema.standardized.dtypes")


# ---------------------------------------------------------------------
# Write block getters
# ---------------------------------------------------------------------


def get_dataset_overwrite_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether outputs should be overwritten."""
    write_cfg = get_dataset_write_config(cfg)
    value = write_cfg.get("overwrite", True)
    return require_bool(value, key="write.overwrite")


def get_dataset_allow_empty_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether empty DataFrames may be written."""
    write_cfg = get_dataset_write_config(cfg)
    value = write_cfg.get("allow_empty", False)
    return require_bool(value, key="write.allow_empty")


# ---------------------------------------------------------------------
# Logging block getters
# ---------------------------------------------------------------------


def get_dataset_logging_level(cfg: dict[str, Any] | None = None) -> str:
    """Return logging level."""
    logging_cfg = get_dataset_logging_config(cfg)
    value = logging_cfg.get("level", "INFO")
    return require_str(value, key="logging.level").strip().upper()


def get_dataset_logging_file(cfg: dict[str, Any] | None = None) -> str | None:
    """Return logging filename."""
    logging_cfg = get_dataset_logging_config(cfg)
    value = logging_cfg.get("file")

    if value is None:
        return None

    return require_str(value, key="logging.file").strip()


def get_dataset_logging_timestamp_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether dataset log filename should be timestamped."""
    logging_cfg = get_dataset_logging_config(cfg)
    value = logging_cfg.get("timestamp", True)
    return require_bool(value, key="logging.timestamp")