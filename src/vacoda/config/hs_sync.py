"""
vacoda.config.hs_sync

Config reader for configs/hs_sync.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vacoda.config.base import (
    CONFIG_DIR,
    get_required_block,
    get_required_nested_block,
    load_yaml_config,
    require_bool,
    require_str,
    require_int,
    require_float,
    require_dict,
)


HS_SYNC_CONFIG_PATH = CONFIG_DIR / "hs_sync.yaml"


def load_hs_sync_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configs/hs_sync.yaml."""
    return load_yaml_config(config_path or HS_SYNC_CONFIG_PATH)


# ---------------------------------------------------------------------
# Scope config getters
# ---------------------------------------------------------------------


def get_scope_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return HS-sync scope config block."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "scope",
        config_name="hs_sync.yaml",
    )


def get_target_hs(cfg: dict[str, Any] | None = None) -> str:
    """Return target HS system, e.g. H5."""
    scope_cfg = get_scope_config(cfg)
    value = scope_cfg.get("target_hs")
    return require_str(value, key="scope.target_hs").strip().upper()


def get_min_year(cfg: dict[str, Any] | None = None) -> int:
    """Return minimum year for HS-sync diagnostics."""
    scope_cfg = get_scope_config(cfg)
    value = scope_cfg.get("min_year", 2002)
    return require_int(value, key="scope.min_year", min_value=1962)


# ---------------------------------------------------------------------
# Logging config getters
# ---------------------------------------------------------------------


def get_logging_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return logging config from hs_sync.yaml."""
    cfg = cfg or load_hs_sync_config()
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


# ---------------------------------------------------------------------
# Engine config getters
# --------------------------------------------------------------------- 

def get_engine_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "engine",
        config_name="hs_sync.yaml",
    )


def get_engine_name(cfg: dict[str, Any] | None = None) -> str:
    engine_cfg = get_engine_config(cfg)
    value = engine_cfg.get("name", "duckdb")
    return require_str(value, key="engine.name").strip().lower()


def get_engine_threads(cfg: dict[str, Any] | None = None) -> int:
    engine_cfg = get_engine_config(cfg)
    value = engine_cfg.get("threads", 16)
    return require_int(value, key="engine.threads", min_value=1)


def get_progress_bool(cfg: dict[str, Any] | None = None) -> bool:
    cfg = cfg or load_hs_sync_config()
    progress_cfg = cfg.get("progress", {})
    value = progress_cfg.get("enabled", True)
    return require_bool(value, key="progress.enabled")

# ------------------------------------------------------------------------------------------------------------------------------------------
# Trade data column getters
# -------------------------------------------------------------------------


def get_columns_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return trade-data columns config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "columns",
        config_name="hs_sync.yaml",
    )


def get_year_col(cfg: dict[str, Any] | None = None) -> str:
    columns_cfg = get_columns_config(cfg)
    value = columns_cfg.get("year", "RefYear")
    return require_str(value, key="columns.year")


def get_hs_col(cfg: dict[str, Any] | None = None) -> str:
    columns_cfg = get_columns_config(cfg)
    value = columns_cfg.get("hs", "ClassificationCode")
    return require_str(value, key="columns.hs")


def get_cmd_col(cfg: dict[str, Any] | None = None) -> str:
    columns_cfg = get_columns_config(cfg)
    value = columns_cfg.get("cmd", "CmdCode")
    return require_str(value, key="columns.cmd")


def get_value_col(cfg: dict[str, Any] | None = None) -> str:
    columns_cfg = get_columns_config(cfg)
    value = columns_cfg.get("value", "PrimaryValue")
    return require_str(value, key="columns.value")


def get_trade_columns(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    """Return standardized trade-data column mapping."""
    return {
        "year": get_year_col(cfg),
        "hs": get_hs_col(cfg),
        "cmd": get_cmd_col(cfg),
        "value": get_value_col(cfg),
    }


# ---------------------------------------------------------------------
# Concordance config getters
# ---------------------------------------------------------------------


def get_concordance_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return concordance config block."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "concordance",
        config_name="hs_sync.yaml",
    )


def get_concordance_stage(cfg: dict[str, Any] | None = None) -> str:
    """Return data stage for concordance file."""
    concordance_cfg = get_concordance_config(cfg)
    value = concordance_cfg.get("stage", "references")
    return require_str(value, key="concordance.stage")


def get_concordance_file(cfg: dict[str, Any] | None = None) -> str:
    """Return concordance filename."""
    concordance_cfg = get_concordance_config(cfg)
    value = concordance_cfg.get("file")

    if not value:
        raise KeyError("Missing required 'concordance.file' in hs_sync.yaml.")

    return require_str(value, key="concordance.file")


def get_concordance_columns_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return concordance column mappings."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("concordance", "columns"),
        config_name="hs_sync.yaml",
    )


def get_concordance_columns(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    """Return standardized concordance column mapping."""
    cols = get_concordance_columns_config(cfg)

    return {
        "source_hs": require_str(
            cols.get("source_hs", "source_hs"),
            key="concordance.columns.source_hs",
        ),
        "source_cmd": require_str(
            cols.get("source_cmd", "source_cmd"),
            key="concordance.columns.source_cmd",
        ),
        "target_hs": require_str(
            cols.get("target_hs", "target_hs"),
            key="concordance.columns.target_hs",
        ),
        "target_cmd": require_str(
            cols.get("target_cmd", "target_cmd"),
            key="concordance.columns.target_cmd",
        ),
        "weight": require_str(
            cols.get("weight", "weight"),
            key="concordance.columns.weight",
        ),
    }


def get_concordance_weight_col(cfg: dict[str, Any] | None = None) -> str:
    """Return concordance weight column name."""
    return get_concordance_columns(cfg)["weight"]


def get_concordance_subdir(cfg: dict[str, Any] | None = None) -> str:
    """Return concordance subdirectory below data stage."""
    concordance_cfg = get_concordance_config(cfg)
    value = concordance_cfg.get("subdir", "")
    return require_str(
        value,
        key="concordance.subdir",
        allow_empty=True,
    )

# ---------------------------------------------------------------------
# Input config getters
# ---------------------------------------------------------------------


def get_input_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return HS-sync input config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "input",
        config_name="hs_sync.yaml",
    )


def get_input_stage(cfg: dict[str, Any] | None = None) -> str:
    input_cfg = get_input_config(cfg)
    value = input_cfg.get("stage", "external")
    return require_str(value, key="input.stage")


def get_input_subdir(cfg: dict[str, Any] | None = None) -> str:
    input_cfg = get_input_config(cfg)
    value = input_cfg.get(
        "subdir",
        "harvard_comtrade/as_reported/raw_parquet_partitioned",
    )
    return require_str(value, key="input.subdir")


def get_input_pattern(cfg: dict[str, Any] | None = None) -> str:
    input_cfg = get_input_config(cfg)
    value = input_cfg.get(
        "pattern",
        "hs_partition=*/year_partition=*/*.parquet",
    )
    return require_str(value, key="input.pattern")

def get_input_source(cfg: dict[str, Any] | None = None) -> str:
    """Return logical input source name for main HS-sync workflow."""
    input_cfg = get_input_config(cfg)
    value = input_cfg.get("source", "harvard_staging")
    return require_str(value, key="input.source").strip().lower()


def get_input_format(cfg: dict[str, Any] | None = None) -> str:
    """Return input file format for main HS-sync workflow."""
    input_cfg = get_input_config(cfg)
    value = input_cfg.get("format", "parquet")
    fmt = require_str(value, key="input.format").strip().lower()

    valid = {"csv", "parquet"}
    if fmt not in valid:
        raise ValueError(
            f"Invalid input.format: {fmt!r}. Valid values: {sorted(valid)}"
        )

    return fmt

# ---------------------------------------------------------------------
# Output config getters
# ---------------------------------------------------------------------


def get_output_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return HS-sync output config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "output",
        config_name="hs_sync.yaml",
    )


def get_output_stage(cfg: dict[str, Any] | None = None) -> str:
    output_cfg = get_output_config(cfg)
    value = output_cfg.get("stage", "interim")
    return require_str(value, key="output.stage")


def get_output_subdir(cfg: dict[str, Any] | None = None) -> str:
    output_cfg = get_output_config(cfg)
    value = output_cfg.get("subdir", "hs_sync")
    return require_str(value, key="output.subdir")


# ---------------------------------------------------------------------
# Write config getters
# ---------------------------------------------------------------------


def get_write_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return output writing preferences."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "write",
        config_name="hs_sync.yaml",
    )


def get_write_csv_bool(cfg: dict[str, Any] | None = None) -> bool:
    write_cfg = get_write_config(cfg)
    value = write_cfg.get("csv", True)
    return require_bool(value, key="write.csv")


def get_write_parquet_bool(cfg: dict[str, Any] | None = None) -> bool:
    write_cfg = get_write_config(cfg)
    value = write_cfg.get("parquet", True)
    return require_bool(value, key="write.parquet")


def get_write_overwrite_bool(cfg: dict[str, Any] | None = None) -> bool:
    write_cfg = get_write_config(cfg)
    value = write_cfg.get("overwrite", True)
    return require_bool(value, key="write.overwrite")


def get_write_timestamp_bool(cfg: dict[str, Any] | None = None) -> bool:
    write_cfg = get_write_config(cfg)
    value = write_cfg.get("timestamp", False)
    return require_bool(value, key="write.timestamp")


# ---------------------------------------------------------------------
# Diagnostics config getters
# ---------------------------------------------------------------------


def get_diagnostics_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return diagnostics config block."""
    cfg = cfg or load_hs_sync_config()
    value = cfg.get("diagnostics", {})
    return require_dict(value, key="diagnostics")


def get_weight_sum_tolerance(cfg: dict[str, Any] | None = None) -> float:
    diagnostics_cfg = get_diagnostics_config(cfg)
    value = diagnostics_cfg.get("weight_sum_tolerance", 0.001)

    # If your base.py does not yet have require_float, either add it there
    # or replace this with float(value).
    return require_float(
        value,
        key="diagnostics.weight_sum_tolerance",
        min_value=0.0,
    )


def get_missing_mapping_top_n(cfg: dict[str, Any] | None = None) -> int:
    diagnostics_cfg = get_diagnostics_config(cfg)
    value = diagnostics_cfg.get("missing_mapping_top_n", 50)
    return require_int(
        value,
        key="diagnostics.missing_mapping_top_n",
        min_value=1,
    )


# ---------------------------------------------------------------------
# Product config getters
# ---------------------------------------------------------------------


def get_products_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return config for all HS-sync products."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "products",
        config_name="hs_sync.yaml",
    )


def get_product_config(
    product_name: str,
    cfg: dict[str, Any] | None = None,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Return config for a specific HS-sync product."""
    products = get_products_config(cfg)

    if product_name not in products:
        if required:
            raise KeyError(
                f"Unknown HS-sync product '{product_name}' in hs_sync.yaml."
            )
        return {}

    return require_dict(
        products[product_name],
        key=f"products.{product_name}",
    )


def is_product_enabled(
    product_name: str,
    cfg: dict[str, Any] | None = None,
    *,
    default: bool = False,
) -> bool:
    """Return whether an HS-sync product is enabled."""
    product_cfg = get_product_config(product_name, cfg)
    value = product_cfg.get("enabled", default)
    return require_bool(value, key=f"products.{product_name}.enabled")


def get_product_filename(
    product_name: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return output filename for an HS-sync product."""
    product_cfg = get_product_config(product_name, cfg)

    filename = product_cfg.get("filename")

    if not filename:
        raise KeyError(
            f"Missing required 'filename' for HS-sync product '{product_name}'."
        )

    return require_str(filename, key=f"products.{product_name}.filename")


def get_product_dependency(
    product_name: str,
    cfg: dict[str, Any] | None = None,
) -> str | None:
    """Return product dependency, if any."""
    product_cfg = get_product_config(product_name, cfg)
    value = product_cfg.get("depends_on")

    if value is None:
        return None

    return require_str(value, key=f"products.{product_name}.depends_on")


def get_enabled_products(cfg: dict[str, Any] | None = None) -> list[str]:
    """Return enabled HS-sync product names."""
    products = get_products_config(cfg)

    return [
        product_name
        for product_name in products
        if is_product_enabled(product_name, cfg)
    ]

def get_product_write_bool(
    product_name: str,
    cfg: dict[str, Any] | None = None,
    *,
    default: bool = True,
) -> bool:
    product_cfg = get_product_config(product_name, cfg)
    value = product_cfg.get("write", default)
    return require_bool(value, key=f"products.{product_name}.write")


def get_product_options(
    product_name: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    product_cfg = get_product_config(product_name, cfg)
    value = product_cfg.get("options", {})
    return require_dict(value, key=f"products.{product_name}.options")

# ---------------------------------------------------------------------
# Main HS-sync report config getters
# ---------------------------------------------------------------------


def get_report_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return main HS-sync report config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "report",
        config_name="hs_sync.yaml",
    )


def get_report_output_kind(cfg: dict[str, Any] | None = None) -> str:
    """Return output kind for main HS-sync reports."""
    report_cfg = get_report_config(cfg)
    value = report_cfg.get("output_kind", "reports")
    return require_str(value, key="report.output_kind").strip().lower()


def get_concordance_coverage_report_filename(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return filename for main HS-sync concordance coverage report."""
    report_cfg = get_report_config(cfg)
    value = report_cfg.get(
        "concordance_coverage_filename",
        "audit_hs_sync_concordance_coverage.txt",
    )
    return require_str(value, key="report.concordance_coverage_filename")


def get_report_timestamp_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether main HS-sync reports should be timestamped."""
    report_cfg = get_report_config(cfg)
    value = report_cfg.get("timestamp", True)
    return require_bool(value, key="report.timestamp")


def get_report_write_latest_bool(cfg: dict[str, Any] | None = None) -> bool:
    """Return whether latest main HS-sync report copies should be written."""
    report_cfg = get_report_config(cfg)
    value = report_cfg.get("write_latest", True)
    return require_bool(value, key="report.write_latest")


# ---------------------------------------------------------------------
# Harvard staging config getters
# ---------------------------------------------------------------------

def get_harvard_staging_input_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard staging input config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_staging", "input"),
        config_name="hs_sync.yaml",
    )


def get_harvard_staging_input_stage(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return input data stage for Harvard staging."""
    input_cfg = get_harvard_staging_input_config(cfg)
    value = input_cfg.get("stage", "raw")
    return require_str(value, key="harvard_staging.input.stage")


def get_harvard_staging_input_subdir(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return input subdirectory for Harvard staging."""
    input_cfg = get_harvard_staging_input_config(cfg)
    value = input_cfg.get("subdir", "api/single")
    return require_str(value, key="harvard_staging.input.subdir")


def get_harvard_staging_input_pattern(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return input glob pattern for Harvard staging."""
    input_cfg = get_harvard_staging_input_config(cfg)
    value = input_cfg.get("pattern", "*.csv")
    return require_str(value, key="harvard_staging.input.pattern")

def get_harvard_staging_output_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard staging output config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_staging", "output"),
        config_name="hs_sync.yaml",
    )


def get_harvard_staging_output_stage(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return output data stage for Harvard staging."""
    output_cfg = get_harvard_staging_output_config(cfg)
    value = output_cfg.get("stage", "external")
    return require_str(value, key="harvard_staging.output.stage")


def get_harvard_staging_output_subdir(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return output subdirectory for Harvard staging."""
    output_cfg = get_harvard_staging_output_config(cfg)
    value = output_cfg.get("subdir", "harvard_comtrade")
    return require_str(value, key="harvard_staging.output.subdir")


# ---------------------------------------------------------------------
# Harvard staging report config getters
# ---------------------------------------------------------------------


def get_harvard_staging_report_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard staging report config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_staging", "report"),
        config_name="hs_sync.yaml",
    )


def get_harvard_staging_report_output_kind(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return output kind for Harvard staging reports."""
    report_cfg = get_harvard_staging_report_config(cfg)
    value = report_cfg.get("output_kind", "reports")
    return require_str(
        value,
        key="harvard_staging.report.output_kind",
    ).strip().lower()


def get_harvard_staging_inventory_filename(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return filename for Harvard staging inventory audit."""
    report_cfg = get_harvard_staging_report_config(cfg)
    value = report_cfg.get(
        "inventory_filename",
        "audit_hs_sync_harvard_staging_inventory.csv",
    )
    return require_str(
        value,
        key="harvard_staging.report.inventory_filename",
    )


def get_harvard_staging_preservation_filename(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return filename for Harvard staging preservation audit."""
    report_cfg = get_harvard_staging_report_config(cfg)
    value = report_cfg.get(
        "preservation_filename",
        "audit_hs_sync_harvard_staging_preservation.csv",
    )
    return require_str(
        value,
        key="harvard_staging.report.preservation_filename",
    )


def get_harvard_staging_report_timestamp_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether Harvard staging reports should be timestamped."""
    report_cfg = get_harvard_staging_report_config(cfg)
    value = report_cfg.get("timestamp", True)
    return require_bool(
        value,
        key="harvard_staging.report.timestamp",
    )


def get_harvard_staging_report_write_latest_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether latest Harvard staging report copies should be written."""
    report_cfg = get_harvard_staging_report_config(cfg)
    value = report_cfg.get("write_latest", True)
    return require_bool(
        value,
        key="harvard_staging.report.write_latest",
    )

# ---------------------------------------------------------------------
# Harvard staging diagnostics config getters
# ---------------------------------------------------------------------
def get_harvard_staging_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard staging config block."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "harvard_staging",
        config_name="hs_sync.yaml",
    )

def get_harvard_staging_enabled_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether Harvard staging is enabled."""
    staging_cfg = get_harvard_staging_config(cfg)
    value = staging_cfg.get("enabled", True)
    return require_bool(value, key="harvard_staging.enabled")

def get_harvard_staging_overwrite_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether Harvard staging should overwrite existing output."""
    staging_cfg = get_harvard_staging_config(cfg)
    value = staging_cfg.get("overwrite", True)
    return require_bool(value, key="harvard_staging.overwrite")

def get_harvard_staging_diagnostics_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard staging diagnostics config."""
    staging_cfg = get_harvard_staging_config(cfg)
    value = staging_cfg.get("diagnostics", {})
    return require_dict(value, key="harvard_staging.diagnostics")


def get_harvard_staging_value_abs_tolerance(
    cfg: dict[str, Any] | None = None,
) -> float:
    """Return absolute tolerance for staging value preservation audit."""
    diagnostics_cfg = get_harvard_staging_diagnostics_config(cfg)
    value = diagnostics_cfg.get("value_abs_tolerance", 1.0)
    return require_float(
        value,
        key="harvard_staging.diagnostics.value_abs_tolerance",
        min_value=0.0,
    )


def get_harvard_staging_value_rel_tolerance(
    cfg: dict[str, Any] | None = None,
) -> float:
    """Return relative tolerance for staging value preservation audit."""
    diagnostics_cfg = get_harvard_staging_diagnostics_config(cfg)
    value = diagnostics_cfg.get("value_rel_tolerance", 1e-10)
    return require_float(
        value,
        key="harvard_staging.diagnostics.value_rel_tolerance",
        min_value=0.0,
    )
# ---------------------------------------------------------------------
# Harvard weights config getters
# ---------------------------------------------------------------------


def get_harvard_weights_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard weight-generation config block."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "harvard_weights",
        config_name="hs_sync.yaml",
    )


def get_harvard_weights_enabled_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether Harvard weight generation is enabled."""
    weights_cfg = get_harvard_weights_config(cfg)
    value = weights_cfg.get("enabled", True)
    return require_bool(value, key="harvard_weights.enabled")

def get_harvard_weights_input_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard weights input config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_weights", "input"),
        config_name="hs_sync.yaml",
    )


def get_harvard_weights_input_stage(
    cfg: dict[str, Any] | None = None,
) -> str:
    input_cfg = get_harvard_weights_input_config(cfg)
    value = input_cfg.get("stage", "external")
    return require_str(value, key="harvard_weights.input.stage")


def get_harvard_weights_input_subdir(
    cfg: dict[str, Any] | None = None,
) -> str:
    input_cfg = get_harvard_weights_input_config(cfg)
    value = input_cfg.get(
        "subdir",
        "harvard_comtrade/as_reported/raw_parquet_partitioned",
    )
    return require_str(value, key="harvard_weights.input.subdir")


def get_harvard_weights_input_pattern(
    cfg: dict[str, Any] | None = None,
) -> str:
    input_cfg = get_harvard_weights_input_config(cfg)
    value = input_cfg.get(
        "pattern",
        "hs_partition=*/year_partition=*/*.parquet",
    )
    return require_str(value, key="harvard_weights.input.pattern")

def get_harvard_weights_static_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard static input config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_weights", "static"),
        config_name="hs_sync.yaml",
    )


def get_harvard_weights_static_stage(
    cfg: dict[str, Any] | None = None,
) -> str:
    static_cfg = get_harvard_weights_static_config(cfg)
    value = static_cfg.get("stage", "references")
    return require_str(value, key="harvard_weights.static.stage")


def get_harvard_weights_static_subdir(
    cfg: dict[str, Any] | None = None,
) -> str:
    static_cfg = get_harvard_weights_static_config(cfg)
    value = static_cfg.get("subdir", "harvard_comtrade")
    return require_str(value, key="harvard_weights.static.subdir")


def get_harvard_weights_static_concordance_file(
    cfg: dict[str, Any] | None = None,
) -> str:
    static_cfg = get_harvard_weights_static_config(cfg)
    value = static_cfg.get("concordance_file")
    return require_str(
        value,
        key="harvard_weights.static.concordance_file",
    )

def get_harvard_weights_output_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard weights output config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_weights", "output"),
        config_name="hs_sync.yaml",
    )


def get_harvard_weights_output_stage(
    cfg: dict[str, Any] | None = None,
) -> str:
    output_cfg = get_harvard_weights_output_config(cfg)
    value = output_cfg.get("stage", "references")
    return require_str(value, key="harvard_weights.output.stage")


def get_harvard_weights_output_subdir(
    cfg: dict[str, Any] | None = None,
) -> str:
    output_cfg = get_harvard_weights_output_config(cfg)
    value = output_cfg.get("subdir", "hs_sync")
    return require_str(value, key="harvard_weights.output.subdir")


def get_harvard_weights_output_file(
    cfg: dict[str, Any] | None = None,
) -> str:
    output_cfg = get_harvard_weights_output_config(cfg)
    value = output_cfg.get("file", "harvard_hs_conversion_weights.parquet")
    return require_str(value, key="harvard_weights.output.file")

def get_harvard_weights_report_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard weights report config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_weights", "report"),
        config_name="hs_sync.yaml",
    )


def get_harvard_weights_report_output_kind(
    cfg: dict[str, Any] | None = None,
) -> str:
    report_cfg = get_harvard_weights_report_config(cfg)
    value = report_cfg.get("output_kind", "reports")
    return require_str(
        value,
        key="harvard_weights.report.output_kind",
    ).strip().lower()


def get_harvard_weights_report_filename(
    cfg: dict[str, Any] | None = None,
) -> str:
    report_cfg = get_harvard_weights_report_config(cfg)
    value = report_cfg.get("filename", "audit_hs_sync_harvard_weights.csv")
    return require_str(value, key="harvard_weights.report.filename")


def get_harvard_weights_report_timestamp_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    report_cfg = get_harvard_weights_report_config(cfg)
    value = report_cfg.get("timestamp", True)
    return require_bool(value, key="harvard_weights.report.timestamp")


def get_harvard_weights_report_write_latest_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    report_cfg = get_harvard_weights_report_config(cfg)
    value = report_cfg.get("write_latest", True)
    return require_bool(value, key="harvard_weights.report.write_latest")