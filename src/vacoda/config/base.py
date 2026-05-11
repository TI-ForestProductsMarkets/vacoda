"""
vacoda.config.base

Shared configuration utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def get_project_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT


def get_config_dir() -> Path:
    """Return the config directory."""
    return CONFIG_DIR


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    if not isinstance(cfg, dict):
        raise TypeError(
            f"Expected YAML config to contain a dict, got {type(cfg).__name__}: {path}"
        )

    return cfg


def as_path(value: str | Path) -> Path:
    """
    Convert config value to Path.

    Absolute paths remain absolute.
    Relative paths are resolved relative to the project root.
    """
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

