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


@dataclass(frozen=True)
class TemplateContext:
    target_hs: str

    @property
    def target_hs_lower(self) -> str:
        return self.target_hs.lower()

    def render(self, value: str) -> str:
        return value.format(
            target_hs=self.target_hs,
            target_hs_lower=self.target_hs_lower,
        )


@dataclass(frozen=True)
class StagePathConfig:
    stage: str
    subdir: str = ""

    def resolve_data_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

@dataclass(frozen=True)
class ProductConfig:
    name: str
    enabled: bool = False
    filename: str | None = None
    depends_on: str | None = None
    write: bool = True
    output_kind: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class HsSyncConfig:
    raw: dict[str, Any]
    scope: ScopeConfig
    engine: EngineConfig
    columns: ColumnConfig
    coverage: CoverageConfig
    products: dict[str, ProductConfig]

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> "HsSyncConfig":
        raw = load_hs_sync_config(config_path)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HsSyncConfig":
        scope = ScopeConfig.from_dict(get_scope_config(raw))
        engine = EngineConfig.from_dict(get_engine_config(raw))
        columns = ColumnConfig.from_dict(get_columns_config(raw))
        coverage = CoverageConfig.from_dict(get_concordance_config(raw))

        products_cfg = get_products_config(raw)
        products = {
            name: ProductConfig(
                name=name,
                enabled=is_product_enabled(name, raw),
                filename=get_product_filename(name, raw) if is_product_enabled(name, raw) else None,
                depends_on=get_product_dependency(name, raw),
                write=get_product_write_bool(name, raw),
                output_kind=get_product_options(name, raw).get("output_kind"),
                options=get_product_options(name, raw),
            )
            for name in products_cfg
        }

        return cls(
            raw=raw,
            scope=scope,
            engine=engine,
            columns=columns,
            coverage=coverage,
            products=products,
        )   






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


def get_target_h5_mapping_interpretation_display_report_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return product config for interpretation display report."""
    return get_product_config(
        "target_h5_mapping_interpretation_display_report",
        cfg,
        required=True,
    )


def get_target_h5_mapping_interpretation_display_report_enabled_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether interpretation display report product is enabled."""
    return is_product_enabled(
        "target_h5_mapping_interpretation_display_report",
        cfg,
        default=False,
    )


def get_target_h5_mapping_interpretation_display_report_kind(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return product kind for interpretation display report."""
    product_cfg = get_target_h5_mapping_interpretation_display_report_config(cfg)
    value = product_cfg.get("kind", "derived")
    return require_str(
        value,
        key="products.target_h5_mapping_interpretation_display_report.kind",
    )


def get_target_h5_mapping_interpretation_display_report_dependency(
    cfg: dict[str, Any] | None = None,
) -> str | None:
    """Return dependency for interpretation display report product."""
    return get_product_dependency(
        "target_h5_mapping_interpretation_display_report",
        cfg,
    )


def get_target_h5_mapping_interpretation_display_report_filename(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return filename for interpretation display report product."""
    return get_product_filename(
        "target_h5_mapping_interpretation_display_report",
        cfg,
    )


def get_target_h5_mapping_interpretation_display_report_write_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether interpretation display report product should be written."""
    return get_product_write_bool(
        "target_h5_mapping_interpretation_display_report",
        cfg,
        default=True,
    )


def get_target_h5_mapping_interpretation_display_report_output_kind(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return output kind for interpretation display report product."""
    product_cfg = get_target_h5_mapping_interpretation_display_report_config(cfg)
    value = require_str(
        product_cfg.get("output_kind", "reports"),
        key="products.target_h5_mapping_interpretation_display_report.output_kind",
    ).strip().lower()

    # Accept singular alias while normalizing to paths.yaml key.
    if value == "report":
        return "reports"

    return value


def get_target_h5_source_code_coverage_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return product config for target-H5 source code coverage."""
    return get_product_config(
        "target_h5_source_code_coverage",
        cfg,
        required=True,
    )


def get_target_h5_source_code_coverage_enabled_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether target-H5 source code coverage product is enabled."""
    return is_product_enabled(
        "target_h5_source_code_coverage",
        cfg,
        default=False,
    )


def get_target_h5_source_code_coverage_kind(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return product kind for target-H5 source code coverage."""
    product_cfg = get_target_h5_source_code_coverage_config(cfg)
    value = product_cfg.get("kind", "derived")
    return require_str(
        value,
        key="products.target_h5_source_code_coverage.kind",
    )


def get_target_h5_source_code_coverage_dependency(
    cfg: dict[str, Any] | None = None,
) -> str | None:
    """Return dependency for target-H5 source code coverage product."""
    return get_product_dependency(
        "target_h5_source_code_coverage",
        cfg,
    )


def get_target_h5_source_code_coverage_filename(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return filename for target-H5 source code coverage product."""
    return get_product_filename(
        "target_h5_source_code_coverage",
        cfg,
    )


def get_target_h5_source_code_coverage_write_bool(
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Return whether target-H5 source code coverage product should be written."""
    return get_product_write_bool(
        "target_h5_source_code_coverage",
        cfg,
        default=True,
    )

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


def get_harvard_staging_filters_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard staging filters config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_staging", "filters"),
        config_name="hs_sync.yaml",
    )


def get_harvard_staging_valid_cmd_regex(
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return regex used to keep valid commodity codes during Harvard staging."""
    filters_cfg = get_harvard_staging_filters_config(cfg)
    value = filters_cfg.get("valid_cmd_regex", r"^[0-9]{6}$")
    return require_str(value, key="harvard_staging.filters.valid_cmd_regex")


def get_harvard_staging_hs_years(
    cfg: dict[str, Any] | None = None,
) -> dict[str, list[int]]:
    """Return allowed years by HS system for Harvard staging filters."""
    filters_cfg = get_harvard_staging_filters_config(cfg)
    value = filters_cfg.get("hs_years", {})
    hs_years_cfg = require_dict(value, key="harvard_staging.filters.hs_years")

    normalized: dict[str, list[int]] = {}

    for hs_system, years in hs_years_cfg.items():
        hs_key = require_str(
            hs_system,
            key="harvard_staging.filters.hs_years.<key>",
        ).strip().upper()

        if not isinstance(years, list):
            raise TypeError(
                "'harvard_staging.filters.hs_years.%s' must be a list, got %s."
                % (hs_key, type(years).__name__)
            )

        normalized[hs_key] = [
            require_int(
                year,
                key=f"harvard_staging.filters.hs_years.{hs_key}",
                min_value=0,
            )
            for year in years
        ]

    return normalized

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


def get_harvard_staging_columns_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard staging columns config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_staging", "columns"),
        config_name="hs_sync.yaml",
    )


def get_harvard_staging_year_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_columns_config(cfg)
    value = cols_cfg.get("year", "refYear")
    return require_str(value, key="harvard_staging.columns.year")


def get_harvard_staging_hs_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_columns_config(cfg)
    value = cols_cfg.get("hs", "classificationCode")
    return require_str(value, key="harvard_staging.columns.hs")


def get_harvard_staging_cmd_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_columns_config(cfg)
    value = cols_cfg.get("cmd", "cmdCode")
    return require_str(value, key="harvard_staging.columns.cmd")


def get_harvard_staging_value_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_columns_config(cfg)
    value = cols_cfg.get("value", "primaryValue")
    return require_str(value, key="harvard_staging.columns.value")


def get_harvard_staging_reporter_code_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_columns_config(cfg)
    value = cols_cfg.get("reporter_code", "reporterCode")
    return require_str(value, key="harvard_staging.columns.reporter_code")


def get_harvard_staging_partner_code_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_columns_config(cfg)
    value = cols_cfg.get("partner_code", "partnerCode")
    return require_str(value, key="harvard_staging.columns.partner_code")


def get_harvard_staging_flow_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_columns_config(cfg)
    value = cols_cfg.get("flow", "flowCode")
    return require_str(value, key="harvard_staging.columns.flow")


def get_harvard_staging_quantity_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_columns_config(cfg)
    value = cols_cfg.get("quantity", "netWgt")
    return require_str(value, key="harvard_staging.columns.quantity")


def get_harvard_staging_columns(
    cfg: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return standardized Harvard staging input column mapping."""
    return {
        "year": get_harvard_staging_year_col(cfg),
        "hs": get_harvard_staging_hs_col(cfg),
        "cmd": get_harvard_staging_cmd_col(cfg),
        "value": get_harvard_staging_value_col(cfg),
        "reporter_code": get_harvard_staging_reporter_code_col(cfg),
        "partner_code": get_harvard_staging_partner_code_col(cfg),
        "flow": get_harvard_staging_flow_col(cfg),
        "quantity": get_harvard_staging_quantity_col(cfg),
    }


def get_harvard_staging_country_reference_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard staging country reference config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_nested_block(
        cfg,
        ("harvard_staging", "country_reference"),
        config_name="hs_sync.yaml",
    )


def get_harvard_staging_country_reference_source(
    cfg: dict[str, Any] | None = None,
) -> str:
    ref_cfg = get_harvard_staging_country_reference_config(cfg)
    value = ref_cfg.get("source", "single_api")
    return require_str(value, key="harvard_staging.country_reference.source")


def get_harvard_staging_country_reference_stage(
    cfg: dict[str, Any] | None = None,
) -> str:
    ref_cfg = get_harvard_staging_country_reference_config(cfg)
    value = ref_cfg.get("stage", "raw")
    return require_str(value, key="harvard_staging.country_reference.stage")


def get_harvard_staging_country_reference_subdir(
    cfg: dict[str, Any] | None = None,
) -> str:
    ref_cfg = get_harvard_staging_country_reference_config(cfg)
    value = ref_cfg.get("subdir", "api/single")
    return require_str(value, key="harvard_staging.country_reference.subdir")


def get_harvard_staging_country_reference_pattern(
    cfg: dict[str, Any] | None = None,
) -> str:
    ref_cfg = get_harvard_staging_country_reference_config(cfg)
    value = ref_cfg.get("pattern", "*.csv")
    return require_str(value, key="harvard_staging.country_reference.pattern")


def get_harvard_staging_country_reference_columns_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Harvard staging country reference column config."""
    ref_cfg = get_harvard_staging_country_reference_config(cfg)
    value = ref_cfg.get("columns", {})
    return require_dict(value, key="harvard_staging.country_reference.columns")


def get_harvard_staging_country_reference_reporter_code_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_country_reference_columns_config(cfg)
    value = cols_cfg.get("reporter_code", "ReporterCode")
    return require_str(
        value,
        key="harvard_staging.country_reference.columns.reporter_code",
    )


def get_harvard_staging_country_reference_reporter_iso_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_country_reference_columns_config(cfg)
    value = cols_cfg.get("reporter_iso", "ReporterISO")
    return require_str(
        value,
        key="harvard_staging.country_reference.columns.reporter_iso",
    )


def get_harvard_staging_country_reference_partner_code_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_country_reference_columns_config(cfg)
    value = cols_cfg.get("partner_code", "PartnerCode")
    return require_str(
        value,
        key="harvard_staging.country_reference.columns.partner_code",
    )


def get_harvard_staging_country_reference_partner_iso_col(
    cfg: dict[str, Any] | None = None,
) -> str:
    cols_cfg = get_harvard_staging_country_reference_columns_config(cfg)
    value = cols_cfg.get("partner_iso", "PartnerISO")
    return require_str(
        value,
        key="harvard_staging.country_reference.columns.partner_iso",
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


def get_coverage_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return target-HS mapping coverage config."""
    if cfg is None:
        cfg = load_hs_sync_config()

    return get_required_block(
        cfg,
        "coverage",
        config_name="hs_sync.yaml",
    )


def get_coverage_enabled_bool(cfg: dict[str, Any] | None = None) -> bool:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("enabled", True)
    return require_bool(value, key="coverage.enabled")


def get_coverage_target_hs(cfg: dict[str, Any] | None = None) -> str:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("target_hs", get_target_hs(cfg))
    return require_str(value, key="coverage.target_hs").strip().upper()


def get_coverage_min_year(cfg: dict[str, Any] | None = None) -> int:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("min_year", 2017)
    return require_int(value, key="coverage.min_year", min_value=0)


def get_coverage_max_year(cfg: dict[str, Any] | None = None) -> int:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("max_year", 9999)
    return require_int(value, key="coverage.max_year", min_value=0)

def get_coverage_input_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("input", {})
    return require_dict(value, key="coverage.input")


def get_coverage_input_stage(cfg: dict[str, Any] | None = None) -> str:
    input_cfg = get_coverage_input_config(cfg)
    value = input_cfg.get("stage", "raw")
    return require_str(value, key="coverage.input.stage")


def get_coverage_input_subdir(cfg: dict[str, Any] | None = None) -> str:
    input_cfg = get_coverage_input_config(cfg)
    value = input_cfg.get("subdir", "")
    return require_str(value, key="coverage.input.subdir")


def get_coverage_input_pattern(cfg: dict[str, Any] | None = None) -> str:
    input_cfg = get_coverage_input_config(cfg)
    value = input_cfg.get("pattern", "*.csv")
    return require_str(value, key="coverage.input.pattern")


def get_coverage_input_format(cfg: dict[str, Any] | None = None) -> str:
    input_cfg = get_coverage_input_config(cfg)
    value = input_cfg.get("format", "csv")
    return require_str(value, key="coverage.input.format").strip().lower()


def get_coverage_mappings_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("mappings", {})
    return require_dict(value, key="coverage.mappings")


def get_coverage_mappings_stage(cfg: dict[str, Any] | None = None) -> str:
    mappings_cfg = get_coverage_mappings_config(cfg)
    value = mappings_cfg.get("stage", "references")
    return require_str(value, key="coverage.mappings.stage")


def get_coverage_mappings_subdir(cfg: dict[str, Any] | None = None) -> str:
    mappings_cfg = get_coverage_mappings_config(cfg)
    value = mappings_cfg.get("subdir", "")
    return require_str(value, key="coverage.mappings.subdir")


def get_coverage_mapping_files(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    mappings_cfg = get_coverage_mappings_config(cfg)
    value = mappings_cfg.get("files", {})
    files_cfg = require_dict(value, key="coverage.mappings.files")

    return {
        require_str(key, key="coverage.mappings.files.<key>"): require_str(
            filename,
            key=f"coverage.mappings.files.{key}",
        )
        for key, filename in files_cfg.items()
    }


def get_coverage_mapping_columns_config(
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mappings_cfg = get_coverage_mappings_config(cfg)
    value = mappings_cfg.get("columns", {})
    return require_dict(value, key="coverage.mappings.columns")


def get_coverage_mapping_source_cmd_col(cfg: dict[str, Any] | None = None) -> str:
    cols_cfg = get_coverage_mapping_columns_config(cfg)
    value = cols_cfg.get("source_cmd", "code.source")
    return require_str(value, key="coverage.mappings.columns.source_cmd")


def get_coverage_mapping_target_cmd_col(cfg: dict[str, Any] | None = None) -> str:
    cols_cfg = get_coverage_mapping_columns_config(cfg)
    value = cols_cfg.get("target_cmd", "code.target")
    return require_str(value, key="coverage.mappings.columns.target_cmd")


def get_coverage_mapping_relationship_col(cfg: dict[str, Any] | None = None) -> str:
    cols_cfg = get_coverage_mapping_columns_config(cfg)
    value = cols_cfg.get("relationship", "Relationship")
    return require_str(value, key="coverage.mappings.columns.relationship")


def get_coverage_mapping_adjustment_col(cfg: dict[str, Any] | None = None) -> str:
    cols_cfg = get_coverage_mapping_columns_config(cfg)
    value = cols_cfg.get("adjustment", "adjustment")
    return require_str(value, key="coverage.mappings.columns.adjustment")


def get_coverage_mapping_group_id_col(cfg: dict[str, Any] | None = None) -> str:
    cols_cfg = get_coverage_mapping_columns_config(cfg)
    value = cols_cfg.get("group_id", "group.id")
    return require_str(value, key="coverage.mappings.columns.group_id")


def get_coverage_mapping_columns(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    return {
        "source_cmd": get_coverage_mapping_source_cmd_col(cfg),
        "target_cmd": get_coverage_mapping_target_cmd_col(cfg),
        "relationship": get_coverage_mapping_relationship_col(cfg),
        "adjustment": get_coverage_mapping_adjustment_col(cfg),
        "group_id": get_coverage_mapping_group_id_col(cfg),
    }

def get_coverage_categories_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("categories", {})
    return require_dict(value, key="coverage.categories")


def get_coverage_easy_without_weights_categories(
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    categories_cfg = get_coverage_categories_config(cfg)
    value = categories_cfg.get("easy_without_weights", ["already_target", "1:1", "n:1"])

    if not isinstance(value, list):
        raise TypeError("'coverage.categories.easy_without_weights' must be a list.")

    return [
        require_str(item, key="coverage.categories.easy_without_weights")
        for item in value
    ]


def get_coverage_needs_weights_categories(
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    categories_cfg = get_coverage_categories_config(cfg)
    value = categories_cfg.get("needs_weights", ["1:n", "n:n"])

    if not isinstance(value, list):
        raise TypeError("'coverage.categories.needs_weights' must be a list.")

    return [
        require_str(item, key="coverage.categories.needs_weights")
        for item in value
    ]


def get_coverage_missing_categories(
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    categories_cfg = get_coverage_categories_config(cfg)
    value = categories_cfg.get("missing", ["missing_mapping"])

    if not isinstance(value, list):
        raise TypeError("'coverage.categories.missing' must be a list.")

    return [
        require_str(item, key="coverage.categories.missing")
        for item in value
    ]

def get_coverage_report_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("report", {})
    return require_dict(value, key="coverage.report")


def get_coverage_report_output_kind(cfg: dict[str, Any] | None = None) -> str:
    report_cfg = get_coverage_report_config(cfg)
    value = report_cfg.get("output_kind", "reports")
    return require_str(value, key="coverage.report.output_kind")


def get_coverage_report_filename(cfg: dict[str, Any] | None = None) -> str:
    report_cfg = get_coverage_report_config(cfg)
    value = report_cfg.get(
        "filename",
        "audit_hs_sync_target_h5_mapping_coverage.csv",
    )
    return require_str(value, key="coverage.report.filename")


def get_coverage_report_timestamp_bool(cfg: dict[str, Any] | None = None) -> bool:
    report_cfg = get_coverage_report_config(cfg)
    value = report_cfg.get("timestamp", True)
    return require_bool(value, key="coverage.report.timestamp")


def get_coverage_report_write_latest_bool(cfg: dict[str, Any] | None = None) -> bool:
    report_cfg = get_coverage_report_config(cfg)
    value = report_cfg.get("write_latest", True)
    return require_bool(value, key="coverage.report.write_latest")

def get_coverage_supported_source_hs(
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("supported_source_hs", ["H4", "H5", "H6"])

    if not isinstance(value, list):
        raise TypeError("'coverage.supported_source_hs' must be a list.")

    return [
        require_str(item, key="coverage.supported_source_hs").strip().upper()
        for item in value
    ]

def get_coverage_output_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("output", {})
    return require_dict(value, key="coverage.output")


def get_coverage_output_stage(cfg: dict[str, Any] | None = None) -> str:
    output_cfg = get_coverage_output_config(cfg)
    value = output_cfg.get("stage", "interim")
    return require_str(value, key="coverage.output.stage")


def get_coverage_output_subdir(cfg: dict[str, Any] | None = None) -> str:
    output_cfg = get_coverage_output_config(cfg)
    value = output_cfg.get("subdir", "hs_sync/target_h5_mapping_coverage")
    return require_str(value, key="coverage.output.subdir")


def get_coverage_write_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    coverage_cfg = get_coverage_config(cfg)
    value = coverage_cfg.get("write", {})
    return require_dict(value, key="coverage.write")


def get_coverage_write_csv_bool(cfg: dict[str, Any] | None = None) -> bool:
    write_cfg = get_coverage_write_config(cfg)
    value = write_cfg.get("csv", True)
    return require_bool(value, key="coverage.write.csv")


def get_coverage_write_parquet_bool(cfg: dict[str, Any] | None = None) -> bool:
    write_cfg = get_coverage_write_config(cfg)
    value = write_cfg.get("parquet", True)
    return require_bool(value, key="coverage.write.parquet")


def get_coverage_write_overwrite_bool(cfg: dict[str, Any] | None = None) -> bool:
    write_cfg = get_coverage_write_config(cfg)
    value = write_cfg.get("overwrite", True)
    return require_bool(value, key="coverage.write.overwrite")


def get_coverage_write_timestamp_bool(cfg: dict[str, Any] | None = None) -> bool:
    write_cfg = get_coverage_write_config(cfg)
    value = write_cfg.get("timestamp", False)
    return require_bool(value, key="coverage.write.timestamp")