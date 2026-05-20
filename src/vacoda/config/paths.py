"""
vacoda.config.paths

Resolver for configs/paths.yaml.

This module only handles stable path configuration.
It does not know about pipeline.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vacoda.config.base import (
    CONFIG_DIR,
    get_required_block,
    get_required_value,
    load_yaml_config,
    require_path,
    resolve_placeholders,
)
from vacoda.utils.logger import make_timestamped_log_path


PATHS_CONFIG_NAME = "paths.yaml"
PATHS_CONFIG_PATH = CONFIG_DIR / PATHS_CONFIG_NAME


def validate_paths_config(cfg: dict[str, Any]) -> None:
    """Validate required paths.yaml structure."""
    get_required_value(cfg, ("storage_root",), config_name=PATHS_CONFIG_NAME)

    data_cfg = get_required_block(
        cfg,
        "data",
        config_name=PATHS_CONFIG_NAME,
    )
    output_cfg = get_required_block(
        cfg,
        "output",
        config_name=PATHS_CONFIG_NAME,
    )
    metadata_cfg = get_required_block(
        cfg,
        "metadata",
        config_name=PATHS_CONFIG_NAME,
    )

    for key in ("root", "raw", "external", "interim", "processed", "references"):
        get_required_value(data_cfg, (key,), config_name=f"{PATHS_CONFIG_NAME}: data")

    for key in ("root", "figures", "logs", "reports", "manifests"):
        get_required_value(output_cfg, (key,), config_name=f"{PATHS_CONFIG_NAME}: output")

    get_required_value(metadata_cfg, ("root",), config_name=f"{PATHS_CONFIG_NAME}: metadata")


def load_paths_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load, resolve, and validate configs/paths.yaml."""
    cfg = load_yaml_config(config_path or PATHS_CONFIG_PATH)
    cfg = resolve_placeholders(cfg, config_name="paths.yaml")
    validate_paths_config(cfg)
    return cfg

def get_storage_root(cfg: dict[str, Any] | None = None) -> Path:
    """Return storage root."""
    if cfg is None:
        cfg = load_paths_config()

    value = get_required_value(
        cfg,
        ("storage_root",),
        config_name=PATHS_CONFIG_NAME,
    )
    return require_path(value, key="storage_root")


def get_data_dir(stage: str, cfg: dict[str, Any] | None = None) -> Path:
    """Return a data directory by stage."""
    if cfg is None:
        cfg = load_paths_config()

    data_cfg = get_required_block(
        cfg,
        "data",
        config_name=PATHS_CONFIG_NAME,
    )

    if stage not in data_cfg:
        valid = ", ".join(data_cfg.keys())
        raise KeyError(f"Unknown data stage '{stage}'. Valid stages: {valid}")

    return require_path(data_cfg[stage], key=f"data.{stage}")


def get_output_dir(kind: str, cfg: dict[str, Any] | None = None) -> Path:
    """Return an output directory by kind."""
    if cfg is None:
        cfg = load_paths_config()

    output_cfg = get_required_block(
        cfg,
        "output",
        config_name=PATHS_CONFIG_NAME,
    )

    if kind not in output_cfg:
        valid = ", ".join(output_cfg.keys())
        raise KeyError(f"Unknown output kind '{kind}'. Valid kinds: {valid}")

    return require_path(output_cfg[kind], key=f"output.{kind}")


def get_log_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return configured log directory."""
    return get_output_dir("logs", cfg=cfg)


def get_metadata_paths(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    """
    Return metadata parquet paths from paths.yaml.

    Skips metadata.root.
    """
    if cfg is None:
        cfg = load_paths_config()
    metadata_cfg = cfg.get("metadata", {})

    return {
        key: require_path(value, key=f"metadata.{key}")
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
    if cfg is None:
        cfg = load_paths_config()
    return require_path(cfg["slurm"]["reports"], key="slurm.reports")


def resolve_log_path(
    log_file_name: str | Path | None,
    *,
    timestamp: bool = True,
) -> Path | None:
    """
    Resolve a log filename against the configured output/logs directory.

    The logs directory comes from configs/paths.yaml:

        output:
          logs: /storage/vacoda/output/logs

    The task config should provide only a filename, e.g.:

        LOG_HS_SYNC_COVERAGE.log
    """
    if log_file_name is None:
        return None

    log_path = get_output_dir("logs") / Path(log_file_name)

    if timestamp:
        log_path = make_timestamped_log_path(log_path)

    return log_path