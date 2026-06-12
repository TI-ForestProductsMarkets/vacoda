"""
vacoda.config.base

Shared configuration utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
from copy import deepcopy
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]

## subdirectories under configs/ for different config types
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_STEPS_DIR = CONFIG_DIR / "steps"
CONFIG_RUNS_DIR = CONFIG_DIR / "runs"

_PLACEHOLDER_PATTERN = re.compile(r"{([^{}]+)}")


def get_required_value(
    cfg: dict[str, Any],
    path: tuple[str, ...],
    *,
    config_name: str,) -> Any:
    """Return a required nested config value.

    Unlike get_required_nested_block(), the final value does not need to be a dict.
    """
    current: Any = cfg

    for key in path:
        if not isinstance(current, dict):
            dotted = ".".join(path)
            raise TypeError(
                f"Cannot access '{dotted}' in {config_name}: "
                f"parent before '{key}' is {type(current).__name__}, "
                "not a dictionary."
            )

        if key not in current:
            dotted = ".".join(path)
            raise KeyError(f"Missing required '{dotted}' value in {config_name}.")

        current = current[key]

    return current


def lookup_dotted_key(cfg: dict[str, Any], dotted_key: str, *, config_name: str) -> Any:
    """Look up a dotted key such as 'data.root' in a nested config dictionary."""
    return get_required_value(
        cfg,
        tuple(dotted_key.split(".")),
        config_name=config_name,
    )


def resolve_placeholders(
    cfg: dict[str, Any],
    *,
    config_name: str,
    max_passes: int = 10,
) -> dict[str, Any]:
    """Resolve placeholders like {storage_root} or {data.root} in a config dict."""
    resolved = deepcopy(cfg)

    for _ in range(max_passes):
        before = repr(resolved)
        resolved = _resolve_placeholders_in_value(
            resolved,
            resolved,
            config_name=config_name,
        )

        if repr(resolved) == before:
            _raise_if_unresolved_placeholders(resolved, config_name=config_name)
            return resolved

    raise ValueError(
        f"Could not fully resolve placeholders in {config_name} "
        f"after {max_passes} passes."
    )


def _resolve_placeholders_in_value(
    value: Any,
    root_cfg: dict[str, Any],
    *,
    config_name: str,
) -> Any:
    """Resolve placeholders recursively in dict/list/string values."""
    if isinstance(value, dict):
        return {
            key: _resolve_placeholders_in_value(
                item,
                root_cfg,
                config_name=config_name,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _resolve_placeholders_in_value(
                item,
                root_cfg,
                config_name=config_name,
            )
            for item in value
        ]

    if isinstance(value, str):
        return _resolve_placeholders_in_string(
            value,
            root_cfg,
            config_name=config_name,
        )

    return value


def _resolve_placeholders_in_string(
    value: str,
    root_cfg: dict[str, Any],
    *,
    config_name: str,
) -> str:
    """Resolve placeholders inside one string."""
    resolved = value

    for placeholder in _PLACEHOLDER_PATTERN.findall(value):
        replacement = lookup_dotted_key(
            root_cfg,
            placeholder,
            config_name=config_name,
        )

        if not isinstance(replacement, str):
            replacement = str(replacement)

        resolved = resolved.replace(
            "{" + placeholder + "}",
            replacement,
        )

    return resolved


def _raise_if_unresolved_placeholders(value: Any, *, config_name: str) -> None:
    """Raise if unresolved placeholders remain after resolution."""
    if isinstance(value, dict):
        for item in value.values():
            _raise_if_unresolved_placeholders(item, config_name=config_name)
        return

    if isinstance(value, list):
        for item in value:
            _raise_if_unresolved_placeholders(item, config_name=config_name)
        return

    if isinstance(value, str):
        unresolved = _PLACEHOLDER_PATTERN.findall(value)
        if unresolved:
            raise ValueError(
                f"Unresolved placeholder(s) in {config_name}: {unresolved} "
                f"in value {value!r}"
            )


def require_path(value: Any, *, key: str) -> Path:
    """Require a config value to be path-like and convert it to Path."""
    if not isinstance(value, (str, Path)):
        raise TypeError(
            f"'{key}' must be a string or Path, got {value!r} "
            f"({type(value).__name__})."
        )

    return as_path(value)

def get_project_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT


def get_config_dir() -> Path:
    """Return the config directory."""
    return CONFIG_DIR

def get_config_path(name: str) -> Path:
    """Return path to a top-level YAML config."""
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    return CONFIG_DIR / name

def get_step_config_path(name: str) -> Path:
    """Return path to a step-level YAML config."""
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    return CONFIG_STEPS_DIR / name


def get_run_config_path(name: str) -> Path:
    """Return path to a run-level YAML config."""
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    return CONFIG_RUNS_DIR / name

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

def require_float(
    value: Any,
    *,
    key: str,
    min_value: float | None = None,
    max_value: float | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"'{key}' must be a float, got bool.")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{key}' must be a float-compatible value.") from exc

    if min_value is not None and out < min_value:
        raise ValueError(f"'{key}' must be >= {min_value}, got {out}.")

    if max_value is not None and out > max_value:
        raise ValueError(f"'{key}' must be <= {max_value}, got {out}.")

    return out