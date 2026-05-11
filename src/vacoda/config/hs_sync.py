"""
vacoda.config.hs_sync

Config reader for configs/hs_sync.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vacoda.config.base import CONFIG_DIR, load_yaml_config


HS_SYNC_CONFIG_PATH = CONFIG_DIR / "hs_sync.yaml"


def load_hs_sync_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configs/hs_sync.yaml."""
    return load_yaml_config(config_path or HS_SYNC_CONFIG_PATH)


def get_coverage_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return hs_sync coverage config block."""
    cfg = cfg or load_hs_sync_config()

    if "coverage" not in cfg:
        raise KeyError("Missing required 'coverage' block in hs_sync.yaml")

    return cfg["coverage"]


def get_coverage_engine_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage engine config."""
    coverage = get_coverage_config(cfg)
    return coverage.get("engine", {})


def get_coverage_engine_name(cfg: dict[str, Any] | None = None) -> str:
    """Return selected coverage engine."""
    engine_cfg = get_coverage_engine_config(cfg)
    return engine_cfg.get("name", "duckdb").strip().lower()


def get_coverage_threads(cfg: dict[str, Any] | None = None) -> int:
    """Return thread count for coverage engine."""
    engine_cfg = get_coverage_engine_config(cfg)
    return int(engine_cfg.get("threads", 16))


def get_coverage_input_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage input config."""
    coverage = get_coverage_config(cfg)
    return coverage["input"]


def get_coverage_output_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage output config."""
    coverage = get_coverage_config(cfg)
    return coverage["output"]


def get_coverage_report_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage report config."""
    coverage = get_coverage_config(cfg)
    return coverage.get("report", {})


def get_coverage_columns(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    """Return year/HS column names."""
    coverage = get_coverage_config(cfg)
    return coverage.get(
        "columns",
        {
            "year": "RefYear",
            "hs": "ClassificationCode",
        },
    )

def get_coverage_progress_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return coverage progress tracking enabled flag."""
    coverage = get_coverage_config(cfg)
    return bool(coverage["progress"].get("enabled", True))


def get_coverage_thresholds(cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Return clean/mixed classification thresholds."""
    coverage = get_coverage_config(cfg)
    return coverage.get(
        "thresholds",
        {
            "clean": 95.0,
            "mostly_clean": 90.0,
        },
    )


def get_coverage_write_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return output writing preferences."""
    coverage = get_coverage_config(cfg)
    return coverage.get(
        "write",
        {
            "csv": True,
            "parquet": True,
            "overwrite": True,
        },
    )

def get_logging_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return logging config from hs_sync.yaml."""
    cfg = cfg or load_hs_sync_config()
    return cfg.get("logging", {})


def get_logging_level(cfg: dict[str, Any] | None = None) -> str:
    """Return logging level string."""
    logging_cfg = get_logging_config(cfg)
    return logging_cfg.get("level", "INFO")


def get_logging_file(cfg: dict[str, Any] | None = None) -> str | None:
    """Return configured log file path."""
    logging_cfg = get_logging_config(cfg)
    return logging_cfg.get("file")


def get_logging_timestamp_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether log filename should be timestamped."""
    logging_cfg = get_logging_config(cfg)
    return bool(logging_cfg.get("timestamp", True))