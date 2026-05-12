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


## Logging config getters
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


## Coverage config getters
def get_coverage_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return hs_sync coverage config block."""
    cfg = cfg or load_hs_sync_config()

    if "coverage" not in cfg:
        raise KeyError("Missing required 'coverage' block in hs_sync.yaml")

    return cfg["coverage"]

### ================ Input config getters ================
def get_coverage_input_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage input config."""
    coverage = get_coverage_config(cfg)
    return coverage["input"]

def get_coverage_input_data_pattern(cfg: dict[str, Any] | None = None) -> str:
    """Return glob pattern for coverage input data files."""
    input_cfg = get_coverage_input_config(cfg)
    return input_cfg.get("pattern", "*.csv")

### ================ Output config getters ================
def get_coverage_output_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage output config."""
    coverage = get_coverage_config(cfg)
    return coverage["output"]

### ================ Engine config getters ================
def get_coverage_engine_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage engine config."""
    coverage = get_coverage_config(cfg)
    return coverage.get("engine", {})


def get_coverage_engine_name(cfg: dict[str, Any] | None = None) -> str:
    """Return selected coverage engine name."""
    engine_cfg = get_coverage_engine_config(cfg)
    return engine_cfg.get("name", "duckdb").strip().lower()


def get_coverage_threads(cfg: dict[str, Any] | None = None) -> int:
    """Return thread count for coverage engine."""
    engine_cfg = get_coverage_engine_config(cfg)
    return int(engine_cfg.get("threads", 16))


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
    return bool(coverage["progress"].get("enabled", True))


# def get_coverage_thresholds(cfg: dict[str, Any] | None = None) -> dict[str, float]:
#     """Return clean/mixed classification thresholds."""
#     coverage = get_coverage_config(cfg)
#     return coverage.get(
#         "thresholds",
#         {
#             "clean": 95.0,
#             "mostly_clean": 90.0,
#         },
#     )


### ================ Write config getters ================
def get_coverage_write_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return output writing preferences."""
    coverage = get_coverage_config(cfg)
    return coverage.get(
        "write",
        {
            "csv": True,
            "parquet": True,
            "overwrite": True,
            "timestamp": True,
        },
    )




# def get_coverage_reporter_year_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
#     coverage = get_coverage_config(cfg)
#     return coverage.get("reporter_year", {"enabled": False})


# def get_coverage_reporter_year_enabled(cfg: dict[str, Any] | None = None) -> bool:
#     reporter_cfg = get_coverage_reporter_year_config(cfg)
#     return bool(reporter_cfg.get("enabled", False))




## HS-sync coverage product config getters

def get_coverage_products_config(cfg: dict | None = None) -> dict:
    """Return config for all coverage products."""
    coverage = get_coverage_config(cfg)

    if "products" not in coverage:
        raise KeyError("Missing required 'products' block in hs_sync.yaml")

    return coverage.get("products", {})

def get_coverage_product_config(
    product_name: str,
    cfg: dict | None = None) -> dict:
    """Return config for a specific coverage product."""
    products = get_coverage_products_config(cfg)
    return products.get(product_name, {})


def is_coverage_product_enabled(
    product_name: str,
    cfg: dict | None = None,
    *,
    default: bool = False) -> bool:
    """Return whether a specific coverage product is enabled."""
    product_cfg = get_coverage_product_config(product_name, cfg)
    return bool(product_cfg.get("enabled", default))


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
    return product_cfg.get("thresholds", {})


def get_coverage_product_dependency(
    product_name: str,
    cfg: dict | None = None,) -> str | None:
    """Return name of product that the specified product depends on, if any."""
    product_cfg = get_coverage_product_config(product_name, cfg)
    return product_cfg.get("depends_on")


def get_coverage_product_top_n(
    product_name: str,
    cfg: dict | None = None,
    *,
    default: int = 10,
) -> int:
    """Return top_n setting for a coverage product."""
    product_cfg = get_coverage_product_config(product_name, cfg)
    return int(product_cfg.get("top_n", default))

## Coverage report config getters
def get_coverage_report_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return coverage report config."""
    coverage = get_coverage_config(cfg)

    if "report" not in coverage:
        raise KeyError("Missing required 'report' block in hs_sync.yaml")

    return coverage.get("report", {})


def get_coverage_report_enabled(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether coverage report generation is enabled."""
    report_cfg = get_coverage_report_config(cfg)
    return bool(report_cfg.get("enabled", True))

def get_coverage_report_output_kind(cfg: dict[str, Any] | None = None) -> str:
    """Return output format for coverage report."""
    report_cfg = get_coverage_report_config(cfg)
    return report_cfg.get("output_kind", "reports").strip().lower()

def get_coverage_report_filename(cfg: dict[str, Any] | None = None) -> str:
    """Return filename for coverage report."""
    report_cfg = get_coverage_report_config(cfg)
    filename = report_cfg.get("filename", "audit_hs_system_coverage_report.txt")
    return str(filename)

def get_coverage_report_timestamp_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether coverage report filename should be timestamped."""
    report_cfg = get_coverage_report_config(cfg)
    return bool(report_cfg.get("timestamp", True))

def get_coverage_report_write_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether coverage report should be written to disk."""
    report_cfg = get_coverage_report_config(cfg)
    return bool(report_cfg.get("write_latest", True))
