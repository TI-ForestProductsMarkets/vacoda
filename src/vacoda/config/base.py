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


def get_required_block(
    cfg: dict[str, Any],
    block_name: str,
    *,
    config_name: str,
) -> dict[str, Any]:
    """Return a required top-level config block.

    Parameters
    ----------
    cfg:
        Loaded YAML config dictionary.
    block_name:
        Required top-level block name, e.g. "coverage", "paths", "comtrade".
    config_name:
        Human-readable config file name for error messages.
    """
    if block_name not in cfg:
        raise KeyError(
            f"Missing required '{block_name}' block in {config_name}."
        )

    block = cfg[block_name]

    if not isinstance(block, dict):
        raise TypeError(
            f"'{block_name}' block in {config_name} must be a dictionary, "
            f"got {type(block).__name__}."
        )

    return block

def get_required_nested_block(
    cfg: dict[str, Any],
    path: tuple[str, ...],
    *,
    config_name: str,
) -> dict[str, Any]:
    """Return a required nested config block."""
    current: Any = cfg

    for key in path:
        if not isinstance(current, dict):
            dotted = ".".join(path)
            raise TypeError(
                f"Cannot access '{dotted}' in {config_name}: "
                f"parent is {type(current).__name__}, not a dictionary."
            )

        if key not in current:
            dotted = ".".join(path)
            raise KeyError(
                f"Missing required '{dotted}' block in {config_name}."
            )

        current = current[key]

    if not isinstance(current, dict):
        dotted = ".".join(path)
        raise TypeError(
            f"'{dotted}' block in {config_name} must be a dictionary, "
            f"got {type(current).__name__}."
        )

    return current

def require_bool(value: Any, *, key: str) -> bool:
    """Require a YAML value to be a real boolean."""
    if not isinstance(value, bool):
        raise TypeError(
            f"'{key}' must be true or false, got {value!r} "
            f"({type(value).__name__})."
        )
    return value


def require_str(value: Any, *, key: str, allow_empty: bool = False) -> str:
    """Require a YAML value to be a string."""
    if not isinstance(value, str):
        raise TypeError(
            f"'{key}' must be a string, got {value!r} "
            f"({type(value).__name__})."
        )

    if not allow_empty and not value.strip():
        raise ValueError(f"'{key}' must not be empty.")

    return value


def require_int(
    value: Any,
    *,
    key: str,
    min_value: int | None = None,
) -> int:
    """Require a YAML value to be an integer."""
    if not isinstance(value, int):
        raise TypeError(
            f"'{key}' must be an integer, got {value!r} "
            f"({type(value).__name__})."
        )

    if min_value is not None and value < min_value:
        raise ValueError(
            f"'{key}' must be >= {min_value}, got {value}."
        )

    return value


def require_dict(value: Any, *, key: str) -> dict[str, Any]:
    """Require a YAML value to be a dictionary."""
    if not isinstance(value, dict):
        raise TypeError(
            f"'{key}' must be a dictionary, got {value!r} "
            f"({type(value).__name__})."
        )

    return value