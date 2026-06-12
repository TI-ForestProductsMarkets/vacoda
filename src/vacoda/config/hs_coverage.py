"""
vacoda.config.hs_coverage

Config reader for configs/hs_coverage.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vacoda.config.base import (
    get_step_config_path,
    get_required_block,
    get_required_nested_block,
    load_yaml_config,
    require_bool,
    require_str,
    require_int,
    require_dict,
)

## HS-coverage specific config getters
HS_COVERAGE_CONFIG_PATH = get_step_config_path("hs_coverage")


def load_hs_coverage_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configs/hs_coverage.yaml."""
    return load_yaml_config(config_path or HS_COVERAGE_CONFIG_PATH)


## Logging config getters
def get_logging_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return logging config from hs_coverage.yaml."""
    cfg = cfg or load_hs_coverage_config()
    return cfg.get("logging", {})


def get_logging_level(cfg: dict[str, Any] | None = None) -> str:
    """Return logging level string."""
    logging_cfg = get_logging_config(cfg)
    value = logging_cfg.get("level", "INFO")
    return require_str(value, key="logging.level").strip().upper()

def get_logging_file(cfg: dict[str, Any] | None = None) -> str | None:
    """Return configured log file path."""
    logging_cfg = get_logging_config(cfg)
    value = logging_cfg.get("file")

    if value is None:
        return None

    return require_str(value, key="logging.file")

def get_logging_timestamp_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether log filename should be timestamped."""
    logging_cfg = get_logging_config(cfg)
    value = logging_cfg.get("timestamp", True)
    return require_bool(value, key="logging.timestamp")


## Coverage config getters
def get_coverage_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return hs_coverage coverage config block."""
    if cfg is None:
        cfg = load_hs_coverage_config()

    return get_required_block(
        cfg,
        "coverage",
        config_name="hs_coverage.yaml",
    )

### ================ Input config getters ================
def get_coverage_input_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_hs_coverage_config()

    return get_required_nested_block(
        cfg,
        ("coverage", "input"),
        config_name="hs_coverage.yaml",
    )

def get_coverage_input_data_pattern(cfg: dict[str, Any] | None = None) -> str:
    """Return glob pattern for coverage input data files."""
    input_cfg = get_coverage_input_config(cfg)
    value = input_cfg.get("pattern", "*.csv")
    return require_str(value, key="coverage.input.pattern")


### ================ Output config getters ================
def get_coverage_output_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage output config."""
    if cfg is None:
        cfg = load_hs_coverage_config()

    return get_required_nested_block(
        cfg,
        ("coverage", "output"),
        config_name="hs_coverage.yaml",
    )

### ================ Engine config getters ================
def get_coverage_engine_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage engine config."""
    coverage = get_coverage_config(cfg)
    return get_required_block(
        coverage,
        "engine",
        config_name="hs_coverage.yaml",
    )


def get_coverage_engine_name(cfg: dict[str, Any] | None = None) -> str:
    """Return selected coverage engine name."""
    engine_cfg = get_coverage_engine_config(cfg)
    value = engine_cfg.get("name", "duckdb")
    if not value:
        raise ValueError("coverage.engine.name cannot be empty.")
    return require_str(value, key="coverage.engine.name").strip().lower()


def get_coverage_threads(cfg: dict[str, Any] | None = None) -> int:
    """Return thread count for coverage engine."""
    engine_cfg = get_coverage_engine_config(cfg)
    value = require_int(
        engine_cfg.get("threads", 16),
        key="coverage.engine.threads",
        min_value=1,
    )

    if value > 64:
        raise ValueError(
            "coverage.engine.threads seems unreasonably high. "
            "Please set it to a value appropriate for your system."
        )

    return value

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

### ================ Progress config getters ================
def get_coverage_progress_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return coverage progress tracking enabled flag."""
    coverage = get_coverage_config(cfg)
    progress_cfg = get_required_block(
        coverage,
        "progress",
        config_name="hs_coverage.yaml",
    )
    value = progress_cfg.get("enabled", True)
    return require_bool(value, key="coverage.progress.enabled")


### ================ Write config getters ================
def get_coverage_write_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return output writing preferences."""
    coverage = get_coverage_config(cfg)
    return get_required_block(
        coverage,
        "write",
        config_name="hs_coverage.yaml",
    )

## HS-sync coverage product config getters

def get_coverage_products_config(cfg: dict | None = None) -> dict:
    """Return config for all coverage products."""
    coverage = get_coverage_config(cfg)
    return get_required_block(
        coverage,
        "products",
        config_name="hs_coverage.yaml",
    )

def get_coverage_product_config(
    product_name: str,
    cfg: dict[str, Any] | None = None,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Return config for a specific coverage product."""
    products = get_coverage_products_config(cfg)

    if product_name not in products:
        if required:
            raise KeyError(
                f"Unknown coverage product '{product_name}' in hs_coverage.yaml."
            )
        return {}

    return require_dict(
        products[product_name],
        key=f"coverage.products.{product_name}",
    )


def is_coverage_product_enabled(
    product_name: str,
    cfg: dict[str, Any] | None = None,
    *,
    default: bool = False,
) -> bool:
    product_cfg = get_coverage_product_config(product_name, cfg)
    value = product_cfg.get("enabled", default)
    return require_bool(value, key=f"products.{product_name}.enabled")


def get_coverage_product_filename(
    product_name: str,
    cfg: dict | None = None,
) -> str:
    product_cfg = get_coverage_product_config(product_name, cfg)

    filename = product_cfg.get("filename")

    if not filename:
        raise KeyError(
            f"Missing required 'filename' for coverage product '{product_name}'."
        )

    if not isinstance(filename, str):
        raise TypeError(
            f"'filename' for coverage product '{product_name}' must be a string."
        )

    return filename


def get_coverage_product_columns(
    product_name: str,
    cfg: dict | None = None,
    required: tuple[str, ...] | None = None,
) -> dict[str, str]:
    product_cfg = get_coverage_product_config(product_name, cfg)
    columns = product_cfg.get("columns")

    if columns is None:
        raise KeyError(
            f"Missing 'columns' block for coverage product '{product_name}'."
        )

    if not isinstance(columns, dict):
        raise TypeError(
            f"'columns' for coverage product '{product_name}' must be a dictionary."
        )

    required = required or ()
    missing = [key for key in required if key not in columns]

    if missing:
        raise KeyError(
            f"Missing required column mapping(s) for coverage product "
            f"'{product_name}': {missing}"
        )

    return columns


def get_coverage_product_thresholds(
    product_name: str,
    cfg: dict | None = None,
) -> dict:
    product_cfg = get_coverage_product_config(product_name, cfg)
    value = product_cfg.get("thresholds", {})
    return require_dict(value, key=f"products.{product_name}.thresholds")


def get_coverage_product_dependency(
    product_name: str,
    cfg: dict | None = None,) -> str | None:
    """Return name of product that the specified product depends on, if any."""
    product_cfg = get_coverage_product_config(product_name, cfg)
    value = product_cfg.get("depends_on")
    return require_str(value, key=f"products.{product_name}.depends_on") if value is not None else None


def get_coverage_product_top_n(
    product_name: str,
    cfg: dict | None = None,
    *,
    default: int = 10,
) -> int:
    """Return top_n setting for a coverage product."""
    product_cfg = get_coverage_product_config(product_name, cfg)
    value = product_cfg.get("top_n", default)
    return require_int(value, key=f"products.{product_name}.top_n")

## Coverage report config getters
def get_coverage_report_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage report config."""
    coverage = get_coverage_config(cfg)
    return get_required_block(
        coverage,
        "report",
        config_name="hs_coverage.yaml",
    )


def get_coverage_report_enabled(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether coverage report generation is enabled."""
    report_cfg = get_coverage_report_config(cfg)
    value = report_cfg.get("enabled", True)
    return require_bool(value, key="coverage.report.enabled")

def get_coverage_report_output_kind(cfg: dict[str, Any] | None = None) -> str:
    """Return output format for coverage report."""
    report_cfg = get_coverage_report_config(cfg)
    value = report_cfg.get("output_kind", "reports")
    return require_str(value, key="coverage.report.output_kind").strip().lower()

def get_coverage_report_filename(cfg: dict[str, Any] | None = None) -> str:
    """Return filename for coverage report."""
    report_cfg = get_coverage_report_config(cfg)
    filename = report_cfg.get("filename", "audit_hs_system_coverage_report.txt")
    return require_str(filename, key="coverage.report.filename")

def get_coverage_report_timestamp_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether coverage report filename should be timestamped."""
    report_cfg = get_coverage_report_config(cfg)
    value = report_cfg.get("timestamp", True)
    return require_bool(value, key="coverage.report.timestamp")

def get_coverage_report_write_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether coverage report should be written to disk."""
    report_cfg = get_coverage_report_config(cfg)
    value = report_cfg.get("write_latest", True)
    return require_bool(value, key="coverage.report.write_latest")
