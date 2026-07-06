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
    get_required_value(cfg, ("project_root",), config_name=PATHS_CONFIG_NAME)

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
    repo_cfg = get_required_block(
        cfg,
        "repo",
        config_name=PATHS_CONFIG_NAME,
    )

    for key in ("root", "raw", "external", "interim", "processed", "references"):
        get_required_value(data_cfg, (key,), config_name=f"{PATHS_CONFIG_NAME}: data")

    for key in ("root", "figures", "logs", "reports", "manifests"):
        get_required_value(output_cfg, (key,), config_name=f"{PATHS_CONFIG_NAME}: output")

    get_required_value(metadata_cfg, ("root",), config_name=f"{PATHS_CONFIG_NAME}: metadata")

    for key in ("root", "references", "geography"):
        get_required_value(repo_cfg, (key,), config_name=f"{PATHS_CONFIG_NAME}: repo")


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


def get_project_root(cfg: dict[str, Any] | None = None) -> Path:
    """Return project root."""
    if cfg is None:
        cfg = load_paths_config()

    value = get_required_value(
        cfg,
        ("project_root",),
        config_name=PATHS_CONFIG_NAME,
    )
    return require_path(value, key="project_root")


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


def get_repo_root(cfg: dict[str, Any] | None = None) -> Path:
    """Return repository root directory from paths.yaml repo block."""
    if cfg is None:
        cfg = load_paths_config()

    repo_cfg = get_required_block(
        cfg,
        "repo",
        config_name=PATHS_CONFIG_NAME,
    )
    return require_path(repo_cfg["root"], key="repo.root")


def get_repo_references_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return repository references directory from paths.yaml repo block."""
    if cfg is None:
        cfg = load_paths_config()

    repo_cfg = get_required_block(
        cfg,
        "repo",
        config_name=PATHS_CONFIG_NAME,
    )
    return require_path(repo_cfg["references"], key="repo.references")


def get_repo_geography_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return repository geography directory from paths.yaml repo block."""
    if cfg is None:
        cfg = load_paths_config()

    repo_cfg = get_required_block(
        cfg,
        "repo",
        config_name=PATHS_CONFIG_NAME,
    )
    return require_path(repo_cfg["geography"], key="repo.geography")

def get_geography_cepii_gravity_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return CEPII gravity directory from paths.yaml repo block."""
    if cfg is None:
        cfg = load_paths_config()

    repo_cfg = get_required_block(
        cfg,
        "repo",
        config_name=PATHS_CONFIG_NAME,
    )
    return require_path(repo_cfg["geography"] / "cepii" / "gravity", key="repo.geography.cepii_gravity_2021")

def get_geography_cepii_legacy_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return CEPII legacy directory from paths.yaml repo block."""
    if cfg is None:
        cfg = load_paths_config()

    repo_cfg = get_required_block(
        cfg,
        "repo",
        config_name=PATHS_CONFIG_NAME,
    )
    return require_path(repo_cfg["geography"] / "cepii" / "legacy", key="repo.geography.cepii_legacy")


def get_geography_vacoda_extension_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return vacoda geography extension directory from paths.yaml repo block."""
    if cfg is None:
        cfg = load_paths_config()

    repo_cfg = get_required_block(
        cfg,
        "repo",
        config_name=PATHS_CONFIG_NAME,
    )
    return require_path(repo_cfg["geography"] / "vacoda" / "extension", key="repo.geography.vacoda_extension")

def get_geography_production_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Return production geography directory from paths.yaml repo block."""
    if cfg is None:
        cfg = load_paths_config()

    repo_cfg = get_required_block(
        cfg,
        "repo",
        config_name=PATHS_CONFIG_NAME,
    )
    return require_path(repo_cfg["geography"] / "production", key="repo.geography.production")


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
    timestamp: bool | str | None = True,
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

    logs_root = get_output_dir("logs")
    candidate = Path(log_file_name)

    if candidate.is_absolute():
        log_path = candidate
    else:
        parts = list(candidate.parts)

        # If caller passed "logs/...", avoid duplicating logs/logs/...
        if parts and parts[0].lower() == "logs":
            parts = parts[1:]

        log_path = logs_root / Path(*parts) if parts else logs_root

    if timestamp:
        # Accept bool (legacy) and explicit timestamp strings.
        if isinstance(timestamp, str):
            log_path = make_timestamped_log_path(log_path, timestamp=timestamp)
        else:
            log_path = make_timestamped_log_path(log_path)

    return log_path