# src/vacoda/config/hs_sync.py

"""
vacoda.config.hs_sync

Class-based config reader for configs/hs_sync.yaml.

The YAML file is treated as the contract for the HS-sync layer.
This module converts the raw YAML dictionary into typed, convenient
dataclass objects used by workflow code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vacoda.config.base import (
    CONFIG_DIR,
    load_yaml_config,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_str,
)
from vacoda.config.paths import get_data_dir, get_output_dir


HS_SYNC_CONFIG_PATH = CONFIG_DIR / "hs_sync.yaml"


# ---------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------


def load_hs_sync_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configs/hs_sync.yaml as a raw dictionary."""
    return load_yaml_config(config_path or HS_SYNC_CONFIG_PATH)

_OUTPUT_KIND_ALIASES = {
    "report": "reports",
    "reports": "reports",
    "manifest": "manifests",
    "manifests": "manifests",
    "figure": "figures",
    "figures": "figures",
    "log": "logs",
    "logs": "logs",
}


def _normalize_output_kind(kind: str) -> str:
    return _OUTPUT_KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())

def _block(
    cfg: dict[str, Any],
    key: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    value = cfg.get(key)

    if value is None:
        if required:
            raise KeyError(f"Missing required config block: {key!r}")
        return {}

    return require_dict(value, key=key)


def _nested_block(
    cfg: dict[str, Any],
    path: tuple[str, ...],
    *,
    required: bool = True,
) -> dict[str, Any]:
    current: Any = cfg
    label = ".".join(path)

    for key in path:
        if not isinstance(current, dict) or key not in current:
            if required:
                raise KeyError(f"Missing required config block: {label!r}")
            return {}
        current = current[key]

    return require_dict(current, key=label)


def _optional_str(value: Any, *, key: str) -> str | None:
    if value is None:
        return None
    return require_str(value, key=key)


def _string_dict(value: Any, *, key: str) -> dict[str, str]:
    raw = require_dict(value, key=key)
    return {
        require_str(k, key=f"{key}.<key>"): require_str(v, key=f"{key}.{k}")
        for k, v in raw.items()
    }


# ---------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateContext:
    """Template values available to YAML filenames/subdirs."""

    target_hs: str

    @property
    def target_hs_upper(self) -> str:
        return self.target_hs.upper()

    @property
    def target_hs_lower(self) -> str:
        return self.target_hs.lower()

    def render(self, value: str) -> str:
        """Render YAML templates like {target_hs_lower}."""
        return value.format(
            target_hs=self.target_hs_upper,
            target_hs_upper=self.target_hs_upper,
            target_hs_lower=self.target_hs_lower,
        )


# ---------------------------------------------------------------------
# Generic reusable config blocks
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class StagePathConfig:
    stage: str
    subdir: str = ""
    pattern: str | None = None
    format: str | None = None

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
        default_stage: str | None = None,
        default_subdir: str = "",
        default_pattern: str | None = None,
        default_format: str | None = None,
        context: TemplateContext | None = None,
    ) -> "StagePathConfig":
        stage = require_str(
            cfg.get("stage", default_stage),
            key=f"{key}.stage",
        )

        subdir = require_str(
            cfg.get("subdir", default_subdir),
            key=f"{key}.subdir",
            allow_empty=True,
        )

        if context is not None:
            subdir = context.render(subdir)

        pattern = cfg.get("pattern", default_pattern)
        fmt = cfg.get("format", default_format)

        return cls(
            stage=stage,
            subdir=subdir,
            pattern=_optional_str(pattern, key=f"{key}.pattern"),
            format=_optional_str(fmt, key=f"{key}.format"),
        )

    def resolve_data_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base


@dataclass(frozen=True)
class ArtifactConfig:
    kind: str
    subdir: str = ""
    filename: str | None = None
    timestamp: bool = True
    write_latest: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
        default_kind: str = "report",
        default_filename: str | None = None,
        context: TemplateContext | None = None,
    ) -> "ArtifactConfig":
        kind = require_str(
            cfg.get("kind", default_kind),
            key=f"{key}.kind",
        ).strip().lower()
        kind = _normalize_output_kind(kind)

        subdir = require_str(
            cfg.get("subdir", ""),
            key=f"{key}.subdir",
            allow_empty=True,
        )

        filename = cfg.get("filename", default_filename)
        if filename is not None:
            filename = require_str(filename, key=f"{key}.filename")
            if context is not None:
                filename = context.render(filename)

        if context is not None:
            subdir = context.render(subdir)

        return cls(
            kind=kind,
            subdir=subdir,
            filename=filename,
            timestamp=require_bool(
                cfg.get("timestamp", True),
                key=f"{key}.timestamp",
            ),
            write_latest=require_bool(
                cfg.get("write_latest", True),
                key=f"{key}.write_latest",
            ),
            options=require_dict(
                cfg.get("options", {}),
                key=f"{key}.options",
            ),
        )

    def resolve_output_dir(self) -> Path:
        base = get_output_dir(_normalize_output_kind(self.kind))
        return base / self.subdir if self.subdir else base


@dataclass(frozen=True)
class WriteConfig:
    csv: bool = True
    parquet: bool = False
    overwrite: bool = True
    timestamp: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "WriteConfig":
        return cls(
            csv=require_bool(cfg.get("csv", True), key=f"{key}.csv"),
            parquet=require_bool(
                cfg.get("parquet", False),
                key=f"{key}.parquet",
            ),
            overwrite=require_bool(
                cfg.get("overwrite", True),
                key=f"{key}.overwrite",
            ),
            timestamp=require_bool(
                cfg.get("timestamp", False),
                key=f"{key}.timestamp",
            ),
        )


# ---------------------------------------------------------------------
# Global config blocks
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeConfig:
    target_hs: str
    min_year: int
    max_year: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ScopeConfig":
        return cls(
            target_hs=require_str(
                cfg.get("target_hs"),
                key="scope.target_hs",
            ).strip().upper(),
            min_year=require_int(
                cfg.get("min_year", 1962),
                key="scope.min_year",
                min_value=1962,
            ),
            max_year=require_int(
                cfg.get("max_year", 9999),
                key="scope.max_year",
                min_value=1962,
            ),
        )


@dataclass(frozen=True)
class EngineConfig:
    name: str
    threads: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "EngineConfig":
        return cls(
            name=require_str(
                cfg.get("name", "duckdb"),
                key="engine.name",
            ).strip().lower(),
            threads=require_int(
                cfg.get("threads", 16),
                key="engine.threads",
                min_value=1,
            ),
        )


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ProgressConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled", True),
                key="progress.enabled",
            )
        )


@dataclass(frozen=True)
class ColumnsConfig:
    year: str
    hs: str
    cmd: str
    value: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str = "columns",
        defaults: dict[str, str] | None = None,
    ) -> "ColumnsConfig":
        defaults = defaults or {
            "year": "RefYear",
            "hs": "ClassificationCode",
            "cmd": "CmdCode",
            "value": "PrimaryValue",
        }

        return cls(
            year=require_str(cfg.get("year", defaults["year"]), key=f"{key}.year"),
            hs=require_str(cfg.get("hs", defaults["hs"]), key=f"{key}.hs"),
            cmd=require_str(cfg.get("cmd", defaults["cmd"]), key=f"{key}.cmd"),
            value=require_str(cfg.get("value", defaults["value"]), key=f"{key}.value"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "year": self.year,
            "hs": self.hs,
            "cmd": self.cmd,
            "value": self.value,
        }


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file: str | None = None
    timestamp: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "LoggingConfig":
        return cls(
            level=require_str(
                cfg.get("level", "INFO"),
                key="logging.level",
            ).strip().upper(),
            file=_optional_str(cfg.get("file"), key="logging.file"),
            timestamp=require_bool(
                cfg.get("timestamp", True),
                key="logging.timestamp",
            ),
        )


# ---------------------------------------------------------------------
# Harvard staging
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HarvardStagingFiltersConfig:
    valid_cmd_regex: str
    hs_years: dict[str, list[int]]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardStagingFiltersConfig":
        hs_years_raw = require_dict(
            cfg.get("hs_years", {}),
            key="harvard_staging.filters.hs_years",
        )

        hs_years: dict[str, list[int]] = {}
        for hs_system, years in hs_years_raw.items():
            hs_key = require_str(
                hs_system,
                key="harvard_staging.filters.hs_years.<key>",
            ).strip().upper()

            if not isinstance(years, list):
                raise TypeError(
                    f"harvard_staging.filters.hs_years.{hs_key} must be a list."
                )

            hs_years[hs_key] = [
                require_int(
                    year,
                    key=f"harvard_staging.filters.hs_years.{hs_key}",
                    min_value=0,
                )
                for year in years
            ]

        return cls(
            valid_cmd_regex=require_str(
                cfg.get("valid_cmd_regex", r"^[0-9]{6}$"),
                key="harvard_staging.filters.valid_cmd_regex",
            ),
            hs_years=hs_years,
        )


@dataclass(frozen=True)
class HarvardStagingColumnsConfig:
    year: str
    hs: str
    cmd: str
    value: str
    reporter_code: str
    partner_code: str
    flow: str
    quantity: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardStagingColumnsConfig":
        return cls(
            year=require_str(
                cfg.get("year", "refYear"),
                key="harvard_staging.columns.year",
            ),
            hs=require_str(
                cfg.get("hs", "classificationCode"),
                key="harvard_staging.columns.hs",
            ),
            cmd=require_str(
                cfg.get("cmd", "cmdCode"),
                key="harvard_staging.columns.cmd",
            ),
            value=require_str(
                cfg.get("value", "primaryValue"),
                key="harvard_staging.columns.value",
            ),
            reporter_code=require_str(
                cfg.get("reporter_code", "reporterCode"),
                key="harvard_staging.columns.reporter_code",
            ),
            partner_code=require_str(
                cfg.get("partner_code", "partnerCode"),
                key="harvard_staging.columns.partner_code",
            ),
            flow=require_str(
                cfg.get("flow", "flowCode"),
                key="harvard_staging.columns.flow",
            ),
            quantity=require_str(
                cfg.get("quantity", "netWgt"),
                key="harvard_staging.columns.quantity",
            ),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "year": self.year,
            "hs": self.hs,
            "cmd": self.cmd,
            "value": self.value,
            "reporter_code": self.reporter_code,
            "partner_code": self.partner_code,
            "flow": self.flow,
            "quantity": self.quantity,
        }


@dataclass(frozen=True)
class HarvardStagingCountryReferenceColumnsConfig:
    reporter_code: str
    reporter_iso: str
    partner_code: str
    partner_iso: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "HarvardStagingCountryReferenceColumnsConfig":
        return cls(
            reporter_code=require_str(
                cfg.get("reporter_code", "ReporterCode"),
                key="harvard_staging.country_reference.columns.reporter_code",
            ),
            reporter_iso=require_str(
                cfg.get("reporter_iso", "ReporterISO"),
                key="harvard_staging.country_reference.columns.reporter_iso",
            ),
            partner_code=require_str(
                cfg.get("partner_code", "PartnerCode"),
                key="harvard_staging.country_reference.columns.partner_code",
            ),
            partner_iso=require_str(
                cfg.get("partner_iso", "PartnerISO"),
                key="harvard_staging.country_reference.columns.partner_iso",
            ),
        )


@dataclass(frozen=True)
class HarvardStagingCountryReferenceConfig:
    source: str
    input: StagePathConfig
    columns: HarvardStagingCountryReferenceColumnsConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "HarvardStagingCountryReferenceConfig":
        return cls(
            source=require_str(
                cfg.get("source", "single_api"),
                key="harvard_staging.country_reference.source",
            ),
            input=StagePathConfig.from_dict(
                cfg,
                key="harvard_staging.country_reference",
                default_stage="raw",
                default_subdir="api/single",
                default_pattern="*.csv",
            ),
            columns=HarvardStagingCountryReferenceColumnsConfig.from_dict(
                require_dict(
                    cfg.get("columns", {}),
                    key="harvard_staging.country_reference.columns",
                )
            ),
        )


@dataclass(frozen=True)
class HarvardStagingDiagnosticsConfig:
    value_abs_tolerance: float
    value_rel_tolerance: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardStagingDiagnosticsConfig":
        return cls(
            value_abs_tolerance=require_float(
                cfg.get("value_abs_tolerance", 1.0),
                key="harvard_staging.diagnostics.value_abs_tolerance",
                min_value=0.0,
            ),
            value_rel_tolerance=require_float(
                cfg.get("value_rel_tolerance", 1e-10),
                key="harvard_staging.diagnostics.value_rel_tolerance",
                min_value=0.0,
            ),
        )


@dataclass(frozen=True)
class HarvardStagingArtifactConfig:
    kind: str
    subdir: str
    inventory_filename: str
    preservation_filename: str
    timestamp: bool
    write_latest: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardStagingArtifactConfig":
        return cls(
            kind=require_str(
                cfg.get("kind", "manifest"),
                key="harvard_staging.artifact.kind",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir", ""),
                key="harvard_staging.artifact.subdir",
                allow_empty=True,
            ),
            inventory_filename=require_str(
                cfg.get(
                    "inventory_filename",
                    "audit_hs_sync_harvard_staging_inventory.csv",
                ),
                key="harvard_staging.artifact.inventory_filename",
            ),
            preservation_filename=require_str(
                cfg.get(
                    "preservation_filename",
                    "audit_hs_sync_harvard_staging_preservation.csv",
                ),
                key="harvard_staging.artifact.preservation_filename",
            ),
            timestamp=require_bool(
                cfg.get("timestamp", True),
                key="harvard_staging.artifact.timestamp",
            ),
            write_latest=require_bool(
                cfg.get("write_latest", True),
                key="harvard_staging.artifact.write_latest",
            ),
        )

    def resolve_output_dir(self) -> Path:
        base = get_output_dir(_normalize_output_kind(self.kind))
        return base / self.subdir if self.subdir else base


@dataclass(frozen=True)
class HarvardStagingConfig:
    enabled: bool
    input: StagePathConfig
    output: StagePathConfig
    filters: HarvardStagingFiltersConfig
    columns: HarvardStagingColumnsConfig
    overwrite: bool
    country_reference: HarvardStagingCountryReferenceConfig
    diagnostics: HarvardStagingDiagnosticsConfig
    artifact: HarvardStagingArtifactConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardStagingConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled", False),
                key="harvard_staging.enabled",
            ),
            input=StagePathConfig.from_dict(
                require_dict(cfg.get("input", {}), key="harvard_staging.input"),
                key="harvard_staging.input",
                default_stage="raw",
            ),
            output=StagePathConfig.from_dict(
                require_dict(cfg.get("output", {}), key="harvard_staging.output"),
                key="harvard_staging.output",
                default_stage="external",
            ),
            filters=HarvardStagingFiltersConfig.from_dict(
                require_dict(cfg.get("filters", {}), key="harvard_staging.filters")
            ),
            columns=HarvardStagingColumnsConfig.from_dict(
                require_dict(cfg.get("columns", {}), key="harvard_staging.columns")
            ),
            overwrite=require_bool(
                cfg.get("overwrite", True),
                key="harvard_staging.overwrite",
            ),
            country_reference=HarvardStagingCountryReferenceConfig.from_dict(
                require_dict(
                    cfg.get("country_reference", {}),
                    key="harvard_staging.country_reference",
                )
            ),
            diagnostics=HarvardStagingDiagnosticsConfig.from_dict(
                require_dict(
                    cfg.get("diagnostics", {}),
                    key="harvard_staging.diagnostics",
                )
            ),
            artifact=HarvardStagingArtifactConfig.from_dict(
                require_dict(cfg.get("artifact", {}), key="harvard_staging.artifact")
            ),
        )

# ---------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageMappingColumnsConfig:
    source_cmd: str
    target_cmd: str
    relationship: str
    adjustment: str
    group_id: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CoverageMappingColumnsConfig":
        return cls(
            source_cmd=require_str(
                cfg.get("source_cmd", "code.source"),
                key="coverage.mappings.columns.source_cmd",
            ),
            target_cmd=require_str(
                cfg.get("target_cmd", "code.target"),
                key="coverage.mappings.columns.target_cmd",
            ),
            relationship=require_str(
                cfg.get("relationship", "Relationship"),
                key="coverage.mappings.columns.relationship",
            ),
            adjustment=require_str(
                cfg.get("adjustment", "adjustment"),
                key="coverage.mappings.columns.adjustment",
            ),
            group_id=require_str(
                cfg.get("group_id", "group.id"),
                key="coverage.mappings.columns.group_id",
            ),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "source_cmd": self.source_cmd,
            "target_cmd": self.target_cmd,
            "relationship": self.relationship,
            "adjustment": self.adjustment,
            "group_id": self.group_id,
        }


@dataclass(frozen=True)
class CoverageMappingsConfig:
    stage: str
    subdir: str
    files: dict[str, str]
    columns: CoverageMappingColumnsConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CoverageMappingsConfig":
        return cls(
            stage=require_str(
                cfg.get("stage", "references"),
                key="coverage.mappings.stage",
            ),
            subdir=require_str(
                cfg.get("subdir", ""),
                key="coverage.mappings.subdir",
                allow_empty=True,
            ),
            files=_string_dict(
                cfg.get("files", {}),
                key="coverage.mappings.files",
            ),
            columns=CoverageMappingColumnsConfig.from_dict(
                require_dict(
                    cfg.get("columns", {}),
                    key="coverage.mappings.columns",
                )
            ),
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_files(self) -> dict[str, Path]:
        root = self.resolve_dir()
        return {name: root / filename for name, filename in self.files.items()}


@dataclass(frozen=True)
class CoverageCategoriesConfig:
    easy_without_weights: list[str]
    needs_weights: list[str]
    missing: list[str]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CoverageCategoriesConfig":
        def parse_list(name: str, default: list[str]) -> list[str]:
            value = cfg.get(name, default)
            if not isinstance(value, list):
                raise TypeError(f"coverage.categories.{name} must be a list.")
            return [
                require_str(item, key=f"coverage.categories.{name}")
                for item in value
            ]

        return cls(
            easy_without_weights=parse_list(
                "easy_without_weights",
                ["already_target", "1:1", "n:1"],
            ),
            needs_weights=parse_list("needs_weights", ["1:n", "n:n"]),
            missing=parse_list("missing", ["missing_mapping"]),
        )


@dataclass(frozen=True)
class ProductConfig:
    name: str
    enabled: bool
    kind: str
    filename: str
    write: bool
    depends_on: str | None = None
    output_kind: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
        *,
        context: TemplateContext,
    ) -> "ProductConfig":
        filename = require_str(
            cfg.get("filename"),
            key=f"products.{name}.filename",
        )

        return cls(
            name=name,
            enabled=require_bool(
                cfg.get("enabled", False),
                key=f"products.{name}.enabled",
            ),
            kind=require_str(
                cfg.get("kind", "derived"),
                key=f"products.{name}.kind",
            ).strip().lower(),
            depends_on=_optional_str(
                cfg.get("depends_on"),
                key=f"products.{name}.depends_on",
            ),
            filename=context.render(filename),
            write=require_bool(
                cfg.get("write", True),
                key=f"products.{name}.write",
            ),
            output_kind=_optional_str(
                cfg.get("output_kind"),
                key=f"products.{name}.output_kind",
            ),
            options=require_dict(
                cfg.get("options", {}),
                key=f"products.{name}.options",
            ),
        )

    def resolve_output_dir(self, default_dir: Path) -> Path:
        if self.output_kind is None:
            return default_dir
        return get_output_dir(self.output_kind.strip().lower())

@dataclass(frozen=True)
class CoverageConfig:
    enabled: bool
    target_hs: str
    min_year: int
    max_year: int
    input: StagePathConfig
    mappings: CoverageMappingsConfig
    categories: CoverageCategoriesConfig
    output: StagePathConfig
    write: WriteConfig
    artifact: ArtifactConfig
    products: dict[str, ProductConfig]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        scope: ScopeConfig,
    ) -> "CoverageConfig":
        target_hs = require_str(
            cfg.get("target_hs", scope.target_hs),
            key="coverage.target_hs",
        ).strip().upper()

        context = TemplateContext(target_hs=target_hs)

        return cls(
            enabled=require_bool(
                cfg.get("enabled", False),
                key="coverage.enabled",
            ),
            target_hs=target_hs,
            min_year=require_int(
                cfg.get("min_year", scope.min_year),
                key="coverage.min_year",
                min_value=0,
            ),
            max_year=require_int(
                cfg.get("max_year", scope.max_year),
                key="coverage.max_year",
                min_value=0,
            ),
            input=StagePathConfig.from_dict(
                require_dict(cfg.get("input", {}), key="coverage.input"),
                key="coverage.input",
                default_stage="raw",
                context=context,
            ),
            mappings=CoverageMappingsConfig.from_dict(
                require_dict(cfg.get("mappings", {}), key="coverage.mappings")
            ),
            categories=CoverageCategoriesConfig.from_dict(
                require_dict(cfg.get("categories", {}), key="coverage.categories")
            ),
            output=StagePathConfig.from_dict(
                require_dict(cfg.get("output", {}), key="coverage.output"),
                key="coverage.output",
                default_stage="interim",
                context=context,
            ),
            write=WriteConfig.from_dict(
                require_dict(cfg.get("write", {}), key="coverage.write"),
                key="coverage.write",
            ),
            artifact=ArtifactConfig.from_dict(
                require_dict(cfg.get("artifact", {}), key="coverage.artifact"),
                key="coverage.artifact",
                default_kind="report",
                default_filename="{target_hs_lower}_mapping_coverage_report",
                context=context,
            ),
            products={
                name: ProductConfig.from_dict(name, cfg, context=context)
                for name, cfg in require_dict(cfg.get("products", {}), key="coverage.products").items()
            }
        )


@dataclass(frozen=True)
class FutureBlockConfig:
    enabled: bool
    status: str | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "FutureBlockConfig":
        return cls(
            enabled=require_bool(cfg.get("enabled", False), key=f"{key}.enabled"),
            status=_optional_str(cfg.get("status"), key=f"{key}.status"),
        )




# ---------------------------------------------------------------------
# Harvard weights
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HarvardWeightsStaticConfig:
    stage: str
    subdir: str
    concordance_file: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardWeightsStaticConfig":
        return cls(
            stage=require_str(
                cfg.get("stage", "references"),
                key="harvard_weights.static.stage",
            ),
            subdir=require_str(
                cfg.get("subdir", "harvard_comtrade"),
                key="harvard_weights.static.subdir",
                allow_empty=True,
            ),
            concordance_file=require_str(
                cfg.get("concordance_file"),
                key="harvard_weights.static.concordance_file",
            ),
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_concordance_path(self) -> Path:
        return self.resolve_dir() / self.concordance_file


@dataclass(frozen=True)
class HarvardWeightsOutputConfig:
    stage: str
    subdir: str
    file: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardWeightsOutputConfig":
        return cls(
            stage=require_str(
                cfg.get("stage", "references"),
                key="harvard_weights.output.stage",
            ),
            subdir=require_str(
                cfg.get("subdir", "hs_sync"),
                key="harvard_weights.output.subdir",
                allow_empty=True,
            ),
            file=require_str(
                cfg.get("file", "harvard_hs_conversion_weights.parquet"),
                key="harvard_weights.output.file",
            ),
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.file


@dataclass(frozen=True)
class HarvardWeightsConfig:
    pass


# ---------------------------------------------------------------------
# Main config object
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HsSyncConfig:
    raw: dict[str, Any]

    scope: ScopeConfig
    engine: EngineConfig
    progress: ProgressConfig
    columns: ColumnsConfig
    logging: LoggingConfig

    harvard_staging: HarvardStagingConfig
    coverage: CoverageConfig

    references: dict[str, Any]

    harvard_weights: FutureBlockConfig
    artifact: ArtifactConfig
    un_comtrade_mapping: FutureBlockConfig
    weight_strategy: FutureBlockConfig

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> "HsSyncConfig":
        raw = load_hs_sync_config(config_path)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HsSyncConfig":
        scope = ScopeConfig.from_dict(_block(global_cfg, "scope"))
        context = TemplateContext(
            target_hs=require_str(
                _block(raw, "coverage", required=False).get(
                    "target_hs",
                    scope.target_hs,
                ),
                key="coverage.target_hs",
            ).strip().upper()
        )

        coverage = CoverageConfig.from_dict(
            _block(raw, "coverage"),
            scope=scope,
        )

        products_raw = _block(raw, "products", required=False)
        products = {
            product_name: ProductConfig.from_dict(
                product_name,
                require_dict(
                    product_cfg,
                    key=f"products.{product_name}",
                ),
                context=TemplateContext(target_hs=coverage.target_hs),
            )
            for product_name, product_cfg in products_raw.items()
        }

        return cls(
            raw=raw,
            scope=scope,
            engine=EngineConfig.from_dict(_block(global_cfg, "engine", required=False)),
            progress=ProgressConfig.from_dict(_block(global_cfg, "progress", required=False)),
            columns=ColumnsConfig.from_dict(_block(global_cfg, "columns")),
            logging=LoggingConfig.from_dict(_block(global_cfg, "logging", required=False)),
            harvard_staging=HarvardStagingConfig.from_dict(
                _block(raw, "harvard_staging")
            ),
            harvard_weights=FutureBlockConfig.from_dict(
                _block(raw, "harvard_weights", required=False),
                key="harvard_weights",
            ),
            coverage=coverage,
            diagnostics=DiagnosticsConfig.from_dict(
                _block(raw, "diagnostics", required=False)
            ),
            artifact=ArtifactConfig.from_dict(
                _block(raw, "artifact", required=False),
                key="artifact",
                default_filename="audit_hs_sync_concordance_coverage.txt",
                context=context,
            ),
            products=products,
            un_comtrade_mapping=FutureBlockConfig.from_dict(
                _block(raw, "un_comtrade_mapping", required=False),
                key="un_comtrade_mapping",
            ),
            weight_strategy=FutureBlockConfig.from_dict(
                _block(raw, "weight_strategy", required=False),
                key="weight_strategy",
            ),
        )

    @property
    def target_hs(self) -> str:
        """Effective target HS for the active coverage workflow."""
        return self.coverage.target_hs

    @property
    def template_context(self) -> TemplateContext:
        return TemplateContext(target_hs=self.target_hs)

    def render(self, value: str) -> str:
        return self.template_context.render(value)

    def product(self, name: str) -> ProductConfig:
        try:
            return self.products[name]
        except KeyError as exc:
            raise KeyError(f"Unknown HS-sync product: {name!r}") from exc

    def is_product_enabled(self, name: str, *, default: bool = False) -> bool:
        product = self.products.get(name)
        if product is None:
            return default
        return product.enabled

    def enabled_products(self) -> list[str]:
        return [
            product_name
            for product_name, product in self.products.items()
            if product.enabled
        ]

    def product_filename(self, name: str) -> str:
        return self.product(name).filename

    def product_options(self, name: str) -> dict[str, Any]:
        return self.product(name).options

    def product_write_enabled(self, name: str, *, default: bool = True) -> bool:
        product = self.products.get(name)
        if product is None:
            return default
        return product.write

    def product_dependency(self, name: str) -> str | None:
        return self.product(name).depends_on

    def product_output_dir(self, name: str, *, default_dir: Path) -> Path:
        return self.product(name).resolve_output_dir(default_dir)

    def coverage_output_dir(self) -> Path:
        return self.coverage.output.resolve_data_dir()

    def coverage_input_dir(self) -> Path:
        return self.coverage.input.resolve_data_dir()

    def coverage_mapping_dir(self) -> Path:
        return self.coverage.mappings.resolve_dir()

    def coverage_mapping_files(self) -> dict[str, Path]:
        return self.coverage.mappings.resolve_files()