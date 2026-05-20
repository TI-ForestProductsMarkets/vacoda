"""
vacoda.config.pipeline

Resolver for configs/pipeline.yaml.

This module understands pipeline configuration but does not know
the physical storage paths from paths.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vacoda.config.base import CONFIG_DIR, load_yaml_config


PIPELINE_CONFIG_PATH = CONFIG_DIR / "pipeline.yaml"


def load_pipeline_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configs/pipeline.yaml."""
    return load_yaml_config(config_path or PIPELINE_CONFIG_PATH)


def get_pipeline_block(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Return the pipeline block. 
    This is the root block for pipeline configuration.
    """
    cfg = cfg or load_pipeline_config()
    return cfg.get("pipeline", {})


def get_pipeline_mode(cfg: dict[str, Any] | None = None) -> str:
    """Return pipeline mode: per_file or merged."""
    block = get_pipeline_block(cfg)
    return block.get("mode", "per_file").strip().lower()


def get_pipeline_source(cfg: dict[str, Any] | None = None) -> str:
    """Return pipeline source: api or local."""
    block = get_pipeline_block(cfg)
    return block.get("source", "api").strip().lower()


def get_input_block(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the input block."""
    cfg = cfg or load_pipeline_config()

    if "input" not in cfg:
        raise KeyError("Missing required 'input' block in pipeline.yaml")

    return cfg["input"]


def get_output_block(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the output block."""
    cfg = cfg or load_pipeline_config()

    if "output" not in cfg:
        raise KeyError("Missing required 'output' block in pipeline.yaml")

    return cfg["output"]


def get_block_stage(block: dict[str, Any]) -> str:
    """Return stage from an input/output block."""
    if "stage" not in block:
        raise KeyError("Missing required 'stage' key in pipeline block")

    return block["stage"]


def get_block_subdir(block: dict[str, Any]) -> str:
    """Return optional subdir from an input/output block."""
    return block.get("subdir", "")


def get_block_files(
    block: dict[str, Any],
    default: str | list[str] | None = None,
) -> list[str]:
    """
    Return files list from an input/output block.

    Accepts:
        files: "file.csv"
        files:
          - "file.csv"
          - "*.csv"
    """
    files = block.get("files", default if default is not None else [])

    if isinstance(files, str):
        return [files]

    if isinstance(files, list):
        return files

    raise TypeError(f"'files' must be str or list[str], got {type(files).__name__}")


def get_logging_block(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return logging block."""
    cfg = cfg or load_pipeline_config()
    return cfg.get("logging", {})


def get_logging_level(cfg: dict[str, Any] | None = None) -> str:
    """Return logging level."""
    block = get_logging_block(cfg)
    return block.get("level", "INFO").strip().upper()


def get_logging_file(cfg: dict[str, Any] | None = None) -> str:
    """Return log filename."""
    block = get_logging_block(cfg)
    return block.get("file", "pipeline.log")


def get_external_block(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return external block."""
    cfg = cfg or load_pipeline_config()
    return cfg.get("external", {})