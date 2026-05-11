"""
vacoda.config.paths

Resolver for configs/paths.yaml.

This module only handles stable path configuration.
It does not know about pipeline.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vacoda.config.base import CONFIG_DIR, as_path, load_yaml_config


PATHS_CONFIG_PATH = CONFIG_DIR / "paths.yaml"


def load_paths_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configs/paths.yaml."""
    return load_yaml_config(config_path or PATHS_CONFIG_PATH)


def get_storage_root(cfg: dict[str, Any] | None = None) -> Path:
    """Return storage root."""
    cfg = cfg or load_paths_config()
    return as_path(cfg["storage_root"])


def get_data_dir(stage: str, cfg: dict[str, Any] | None = None) -> Path:
    """
    Return a data directory by stage.

    Examples:
        raw, external, interim, processed, references
    """
    cfg = cfg or load_paths_config()

    try:
        return as_path(cfg["data"][stage])
    except KeyError as exc:
        valid = ", ".join(cfg.get("data", {}).keys())
        raise KeyError(f"Unknown data stage '{stage}'. Valid stages: {valid}") from exc


def get_output_dir(kind: str, cfg: dict[str, Any] | None = None) -> Path:
    """
    Return an output directory by kind.

    Examples:
        figures, logs, reports, manifests
    """
    cfg = cfg or load_paths_config()

    try:
        return as_path(cfg["output"][kind])
    except KeyError as exc:
        valid = ", ".join(cfg.get("output", {}).keys())
        raise KeyError(f"Unknown output kind '{kind}'. Valid kinds: {valid}") from exc


def get_log_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return configured log directory."""
    cfg = cfg or load_paths_config()
    return as_path(cfg["logging"]["dir"])


def get_metadata_paths(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    """
    Return metadata parquet paths from paths.yaml.

    Skips metadata.root.
    """
    cfg = cfg or load_paths_config()
    metadata_cfg = cfg.get("metadata", {})

    return {
        key: as_path(value)
        for key, value in metadata_cfg.items()
        if key != "root"
    }


def get_figure_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return figures directory."""
    return get_output_dir("figures", cfg=cfg)


def get_reports_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return reports directory."""
    return get_output_dir("reports", cfg=cfg)


def get_manifests_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return manifests directory."""
    return get_output_dir("manifests", cfg=cfg)


def get_slurm_reports_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return SLURM reports directory."""
    cfg = cfg or load_paths_config()
    return as_path(cfg["slurm"]["reports"])