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
from vacoda.config.paths import get_data_dir, get_output_dir, get_repo_references_dir


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
    subdir: str = ""
    file: str | None = None
    timestamp: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "LoggingConfig":
        return cls(
            level=require_str(
                cfg.get("level", "INFO"),
                key="logging.level",
            ).strip().upper(),
            subdir=require_str(
                cfg.get("subdir", ""),
                key="logging.subdir",
                allow_empty=True,
            ),
            file=_optional_str(cfg.get("file"), key="logging.file"),
            timestamp=require_bool(
                cfg.get("timestamp", True),
                key="logging.timestamp",
            ),
        )

    def resolve_log_dir(self) -> Path:
        base = get_output_dir("logs")
        if not self.subdir:
            return base

        subdir_path = Path(self.subdir)
        parts = list(subdir_path.parts)

        # Treat "logs/..." as already anchored at output.logs root.
        if parts and parts[0].lower() == "logs":
            parts = parts[1:]

        return base / Path(*parts) if parts else base

    def resolve_log_file(self) -> Path | None:
        if self.file is None:
            return None
        return self.resolve_log_dir() / self.file


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
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        context: TemplateContext,
    ) -> "HarvardStagingArtifactConfig":
        return cls(
            kind=require_str(
                cfg.get("kind", "manifest"),
                key="harvard_staging.artifact.kind",
            ).strip().lower(),
            subdir=context.render(
                require_str(
                    cfg.get("subdir", ""),
                    key="harvard_staging.artifact.subdir",
                    allow_empty=True,
                )
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
class HarvardStagingLayoutConfig:
    base_subdir: str
    partitioned_subdir: str
    raw_parquet_subdir: str
    aggregated_subdir: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardStagingLayoutConfig":
        return cls(
            base_subdir=require_str(
                cfg.get("base_subdir", "as_reported"),
                key="harvard_staging.layout.base_subdir",
            ),
            partitioned_subdir=require_str(
                cfg.get("partitioned_subdir", "raw_parquet_partitioned"),
                key="harvard_staging.layout.partitioned_subdir",
            ),
            raw_parquet_subdir=require_str(
                cfg.get("raw_parquet_subdir", "raw_parquet"),
                key="harvard_staging.layout.raw_parquet_subdir",
            ),
            aggregated_subdir=require_str(
                cfg.get(
                    "aggregated_subdir",
                    "aggregated_by_year_not_converted/parquet",
                ),
                key="harvard_staging.layout.aggregated_subdir",
            ),
        )

    def resolve_base_dir(self, output_root: Path) -> Path:
        return output_root / self.base_subdir

    def resolve_partitioned_dir(self, output_root: Path) -> Path:
        return self.resolve_base_dir(output_root) / self.partitioned_subdir

    def resolve_raw_parquet_dir(self, output_root: Path) -> Path:
        return self.resolve_base_dir(output_root) / self.raw_parquet_subdir

    def resolve_aggregated_dir(self, output_root: Path) -> Path:
        return self.resolve_base_dir(output_root) / self.aggregated_subdir


@dataclass(frozen=True)
class HarvardStagingSkipConfig:
    if_complete: bool
    marker_filename: str
    check_input_fingerprint: bool
    check_config_fingerprint: bool
    check_output_fingerprint: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardStagingSkipConfig":
        return cls(
            if_complete=require_bool(
                cfg.get("if_complete", False),
                key="harvard_staging.skip.if_complete",
            ),
            marker_filename=require_str(
                cfg.get("marker_filename", "harvard_staging_success.json"),
                key="harvard_staging.skip.marker_filename",
            ),
            check_input_fingerprint=require_bool(
                cfg.get("check_input_fingerprint", True),
                key="harvard_staging.skip.check_input_fingerprint",
            ),
            check_config_fingerprint=require_bool(
                cfg.get("check_config_fingerprint", True),
                key="harvard_staging.skip.check_config_fingerprint",
            ),
            check_output_fingerprint=require_bool(
                cfg.get("check_output_fingerprint", True),
                key="harvard_staging.skip.check_output_fingerprint",
            ),
        )

@dataclass(frozen=True)
class HarvardStagingConfig:
    enabled: bool
    input: StagePathConfig
    output: StagePathConfig
    layout: HarvardStagingLayoutConfig
    skip: HarvardStagingSkipConfig
    filters: HarvardStagingFiltersConfig
    columns: HarvardStagingColumnsConfig
    overwrite: bool
    country_reference: HarvardStagingCountryReferenceConfig
    diagnostics: HarvardStagingDiagnosticsConfig
    artifact: HarvardStagingArtifactConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        scope: ScopeConfig,
    ) -> "HarvardStagingConfig":
        context = TemplateContext(target_hs=scope.target_hs)

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
            layout=HarvardStagingLayoutConfig.from_dict(
                require_dict(cfg.get("layout", {}), key="harvard_staging.layout")
            ),
            skip=HarvardStagingSkipConfig.from_dict(
                require_dict(cfg.get("skip", {}), key="harvard_staging.skip")
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
                require_dict(cfg.get("artifact", {}), key="harvard_staging.artifact"),
                context=context,
            ),
        )


# ---------------------------------------------------------------------
# Harvard generator audit
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HarvardGeneratorAuditInputConfig:
    harvard_data: StagePathConfig
    generator_root: StagePathConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "HarvardGeneratorAuditInputConfig":
        return cls(
            harvard_data=StagePathConfig.from_dict(
                require_dict(
                    cfg.get("harvard_data", {}),
                    key="harvard_generator_audit.input.harvard_data",
                ),
                key="harvard_generator_audit.input.harvard_data",
                default_stage="external",
                default_subdir="harvard_comtrade_bulk",
            ),
            generator_root=StagePathConfig.from_dict(
                require_dict(
                    cfg.get("generator_root", {}),
                    key="harvard_generator_audit.input.generator_root",
                ),
                key="harvard_generator_audit.input.generator_root",
                default_stage="references",
                default_subdir="comtrade-conversion-weights/generator",
            ),
        )

    def resolve_harvard_data_dir(self) -> Path:
        return self.harvard_data.resolve_data_dir()

    def resolve_generator_root_dir(self) -> Path:
        # Harvard generator code lives under repository references, not storage references.
        if self.generator_root.stage.strip().lower() == "references":
            base = get_repo_references_dir()
            return base / self.generator_root.subdir if self.generator_root.subdir else base
        return self.generator_root.resolve_data_dir()


@dataclass(frozen=True)
class HarvardGeneratorAuditTransitionConfig:
    source_hs: str
    source_year: int
    target_hs: str
    target_year: int

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "HarvardGeneratorAuditTransitionConfig":
        return cls(
            source_hs=require_str(
                cfg.get("source_hs", "H4"),
                key="harvard_generator_audit.transition.source_hs",
            ).strip().upper(),
            source_year=require_int(
                cfg.get("source_year", 2016),
                key="harvard_generator_audit.transition.source_year",
                min_value=0,
            ),
            target_hs=require_str(
                cfg.get("target_hs", "H5"),
                key="harvard_generator_audit.transition.target_hs",
            ).strip().upper(),
            target_year=require_int(
                cfg.get("target_year", 2017),
                key="harvard_generator_audit.transition.target_year",
                min_value=0,
            ),
        )


@dataclass(frozen=True)
class HarvardGeneratorAuditProductCodeCheck:
    label: str
    hs: str
    year: int
    codes: list[str]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        index: int,
    ) -> "HarvardGeneratorAuditProductCodeCheck":
        key = f"harvard_generator_audit.product_code_checks[{index}]"

        codes_raw = cfg.get("codes", [])
        if not isinstance(codes_raw, list):
            raise TypeError(f"{key}.codes must be a list.")

        return cls(
            label=require_str(
                cfg.get("label", f"check_{index}"),
                key=f"{key}.label",
            ),
            hs=require_str(
                cfg.get("hs"),
                key=f"{key}.hs",
            ).strip().upper(),
            year=require_int(
                cfg.get("year"),
                key=f"{key}.year",
                min_value=0,
            ),
            codes=[
                require_str(code, key=f"{key}.codes")
                for code in codes_raw
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "hs": self.hs,
            "year": self.year,
            "codes": self.codes,
        }


@dataclass(frozen=True)
class HarvardGeneratorAuditConfig:
    enabled: bool
    input: HarvardGeneratorAuditInputConfig
    transition: HarvardGeneratorAuditTransitionConfig
    product_code_checks: list[HarvardGeneratorAuditProductCodeCheck]
    artifact: ArtifactConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        scope: ScopeConfig,
    ) -> "HarvardGeneratorAuditConfig":
        checks_raw = cfg.get("product_code_checks", [])
        if not isinstance(checks_raw, list):
            raise TypeError("harvard_generator_audit.product_code_checks must be a list.")

        context = TemplateContext(target_hs=scope.target_hs)

        return cls(
            enabled=require_bool(
                cfg.get("enabled", False),
                key="harvard_generator_audit.enabled",
            ),
            input=HarvardGeneratorAuditInputConfig.from_dict(
                require_dict(
                    cfg.get("input", {}),
                    key="harvard_generator_audit.input",
                )
            ),
            transition=HarvardGeneratorAuditTransitionConfig.from_dict(
                require_dict(
                    cfg.get("transition", {}),
                    key="harvard_generator_audit.transition",
                )
            ),
            product_code_checks=[
                HarvardGeneratorAuditProductCodeCheck.from_dict(
                    require_dict(item, key=f"harvard_generator_audit.product_code_checks[{i}]"),
                    index=i,
                )
                for i, item in enumerate(checks_raw)
            ],
            artifact=ArtifactConfig.from_dict(
                require_dict(
                    cfg.get("artifact", {}),
                    key="harvard_generator_audit.artifact",
                ),
                key="harvard_generator_audit.artifact",
                default_kind="manifest",
                default_filename="audit_harvard_generator_inputs",
                context=context,
            ),
        )

    def product_code_checks_as_dicts(self) -> list[dict[str, Any]]:
        return [check.as_dict() for check in self.product_code_checks]


@dataclass(frozen=True)
class HarvardMappingDirectionMappingsConfig:
    references_root: str = "repo"

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "HarvardMappingDirectionMappingsConfig":
        references_root = require_str(
            cfg.get("references_root", "repo"),
            key="harvard_mapping_direction.mappings.references_root",
        ).strip().lower()

        if references_root not in {"repo", "data"}:
            raise ValueError(
                "harvard_mapping_direction.mappings.references_root must be "
                "'repo' or 'data'."
            )

        return cls(references_root=references_root)


@dataclass(frozen=True)
class HarvardMappingDirectionConfig:
    enabled: bool
    mappings: HarvardMappingDirectionMappingsConfig
    artifact: ArtifactConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        scope: ScopeConfig,
    ) -> "HarvardMappingDirectionConfig":
        artifact_cfg = require_dict(
            cfg.get("artifact", {}),
            key="harvard_mapping_direction.artifact",
        )
        if "subdir" not in artifact_cfg:
            artifact_cfg = {
                **artifact_cfg,
                "subdir": "hs_sync",
            }

        context = TemplateContext(target_hs=scope.target_hs)

        return cls(
            enabled=require_bool(
                cfg.get("enabled", True),
                key="harvard_mapping_direction.enabled",
            ),
            mappings=HarvardMappingDirectionMappingsConfig.from_dict(
                require_dict(
                    cfg.get("mappings", {}),
                    key="harvard_mapping_direction.mappings",
                )
            ),
            artifact=ArtifactConfig.from_dict(
                artifact_cfg,
                key="harvard_mapping_direction.artifact",
                default_kind="reports",
                default_filename="inspect_hscodes_requiring_weights.md",
                context=context,
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
    references_root: str
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
            references_root=require_str(
                cfg.get("references_root", "data"),
                key="coverage.mappings.references_root",
            ).strip().lower(),
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
        stage = self.stage.strip().lower()
        if stage == "references" and self.references_root == "repo":
            base = get_repo_references_dir()
        else:
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
            options=require_dict(
                cfg.get("options", {}),
                key=f"products.{name}.options",
            ),
        )

    def resolve_output_dir(self, default_dir: Path) -> Path:
        return default_dir

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
        target_hs = scope.target_hs

        # target_hs is governed by global.scope.target_hs.
        # If coverage.target_hs is still present during migration, require equality.
        coverage_target_hs = cfg.get("target_hs")
        if coverage_target_hs is not None:
            parsed = require_str(
                coverage_target_hs,
                key="coverage.target_hs",
            ).strip().upper()
            if parsed != target_hs:
                raise ValueError(
                    "coverage.target_hs must match global.scope.target_hs "
                    f"({target_hs!r}); got {parsed!r}."
                )

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


# ---------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class WeightsStrategyConfig:
    name: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "WeightsStrategyConfig":
        return cls(
            name=require_str(
                cfg.get("name", "equal_split"),
                key="weights.strategy.name",
            ).strip().lower(),
        )


@dataclass(frozen=True)
class WeightsInputConfig:
    coverage_product: str
    mappings: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "WeightsInputConfig":
        return cls(
            coverage_product=require_str(
                cfg.get("coverage_product", "source_code_coverage"),
                key="weights.input.coverage_product",
            ),
            mappings=require_str(
                cfg.get("mappings", "coverage.mappings"),
                key="weights.input.mappings",
            ),
        )


@dataclass(frozen=True)
class WeightsProductConfig:
    name: str
    enabled: bool
    filename: str

    @classmethod
    def from_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
        *,
        context: TemplateContext,
    ) -> "WeightsProductConfig":
        return cls(
            name=name,
            enabled=require_bool(
                cfg.get("enabled", False),
                key=f"weights.products.{name}.enabled",
            ),
            filename=context.render(
                require_str(
                    cfg.get("filename"),
                    key=f"weights.products.{name}.filename",
                )
            ),
        )


@dataclass(frozen=True)
class WeightsArtifactConfig:
    kind: str
    subdir: str
    equal_split_audit_filename: str
    complete_weights_audit_filename: str
    timestamp: bool
    write_latest: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        context: TemplateContext,
    ) -> "WeightsArtifactConfig":
        return cls(
            kind=require_str(
                cfg.get("kind", "manifest"),
                key="weights.artifact.kind",
            ).strip().lower(),
            subdir=context.render(
                require_str(
                    cfg.get("subdir", ""),
                    key="weights.artifact.subdir",
                    allow_empty=True,
                )
            ),
            equal_split_audit_filename=context.render(
                require_str(
                    cfg.get("equal_split_audit_filename"),
                    key="weights.artifact.equal_split_audit_filename",
                )
            ),
            complete_weights_audit_filename=context.render(
                require_str(
                    cfg.get("complete_weights_audit_filename"),
                    key="weights.artifact.complete_weights_audit_filename",
                )
            ),
            timestamp=require_bool(
                cfg.get("timestamp", True),
                key="weights.artifact.timestamp",
            ),
            write_latest=require_bool(
                cfg.get("write_latest", True),
                key="weights.artifact.write_latest",
            ),
        )

    def resolve_output_dir(self) -> Path:
        base = get_output_dir(_normalize_output_kind(self.kind))
        return base / self.subdir if self.subdir else base


@dataclass(frozen=True)
class WeightsConfig:
    enabled: bool
    target_hs: str
    strategy: WeightsStrategyConfig
    input: WeightsInputConfig
    output: StagePathConfig
    write: WriteConfig
    products: dict[str, WeightsProductConfig]
    artifact: WeightsArtifactConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        scope: ScopeConfig,
    ) -> "WeightsConfig":
        target_hs = scope.target_hs

        # target_hs is governed by global.scope.target_hs.
        # If weights.target_hs is still present during migration, require equality.
        weights_target_hs = cfg.get("target_hs")
        if weights_target_hs is not None:
            parsed = require_str(
                weights_target_hs,
                key="weights.target_hs",
            ).strip().upper()
            if parsed != target_hs:
                raise ValueError(
                    "weights.target_hs must match global.scope.target_hs "
                    f"({target_hs!r}); got {parsed!r}."
                )

        context = TemplateContext(target_hs=target_hs)

        return cls(
            enabled=require_bool(
                cfg.get("enabled", False),
                key="weights.enabled",
            ),
            target_hs=target_hs,
            strategy=WeightsStrategyConfig.from_dict(
                require_dict(
                    cfg.get("strategy", {}),
                    key="weights.strategy",
                )
            ),
            input=WeightsInputConfig.from_dict(
                require_dict(
                    cfg.get("input", {}),
                    key="weights.input",
                )
            ),
            output=StagePathConfig.from_dict(
                require_dict(
                    cfg.get("output", {}),
                    key="weights.output",
                ),
                key="weights.output",
                default_stage="interim",
                context=context,
            ),
            write=WriteConfig.from_dict(
                require_dict(
                    cfg.get("write", {}),
                    key="weights.write",
                ),
                key="weights.write",
            ),
            products={
                name: WeightsProductConfig.from_dict(name, item_cfg, context=context)
                for name, item_cfg in require_dict(
                    cfg.get("products", {}),
                    key="weights.products",
                ).items()
            },
            artifact=WeightsArtifactConfig.from_dict(
                require_dict(
                    cfg.get("artifact", {}),
                    key="weights.artifact",
                ),
                context=context,
            ),
        )


@dataclass(frozen=True)
class ApplyInputConfig:
    trade_data: str
    weights: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ApplyInputConfig":
        return cls(
            trade_data=require_str(
                cfg.get("trade_data", "coverage.input"),
                key="apply.input.trade_data",
            ),
            weights=require_str(
                cfg.get("weights", "weights.complete_mapping_weights"),
                key="apply.input.weights",
            ),
        )


@dataclass(frozen=True)
class ApplyWriteConfig:
    format: str
    overwrite: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ApplyWriteConfig":
        return cls(
            format=require_str(
                cfg.get("format", "parquet"),
                key="apply.write.format",
            ).strip().lower(),
            overwrite=require_bool(
                cfg.get("overwrite", True),
                key="apply.write.overwrite",
            ),
        )


@dataclass(frozen=True)
class ApplyProductConfig:
    name: str
    enabled: bool
    filename: str

    @classmethod
    def from_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
        *,
        context: TemplateContext,
    ) -> "ApplyProductConfig":
        return cls(
            name=name,
            enabled=require_bool(
                cfg.get("enabled", False),
                key=f"apply.products.{name}.enabled",
            ),
            filename=context.render(
                require_str(
                    cfg.get("filename"),
                    key=f"apply.products.{name}.filename",
                )
            ),
        )


@dataclass(frozen=True)
class ApplyArtifactConfig:
    kind: str
    subdir: str
    audit_filename: str
    integrity_audit_filename: str
    timestamp: bool
    write_latest: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        context: TemplateContext,
    ) -> "ApplyArtifactConfig":
        return cls(
            kind=require_str(
                cfg.get("kind", "manifest"),
                key="apply.artifact.kind",
            ).strip().lower(),
            subdir=context.render(
                require_str(
                    cfg.get("subdir", ""),
                    key="apply.artifact.subdir",
                    allow_empty=True,
                )
            ),
            audit_filename=context.render(
                require_str(
                    cfg.get("audit_filename"),
                    key="apply.artifact.audit_filename",
                )
            ),
            integrity_audit_filename=context.render(
                require_str(
                    cfg.get("integrity_audit_filename"),
                    key="apply.artifact.integrity_audit_filename",
                )
            ),
            timestamp=require_bool(
                cfg.get("timestamp", True),
                key="apply.artifact.timestamp",
            ),
            write_latest=require_bool(
                cfg.get("write_latest", True),
                key="apply.artifact.write_latest",
            ),
        )

    def resolve_output_dir(self) -> Path:
        base = get_output_dir(_normalize_output_kind(self.kind))
        return base / self.subdir if self.subdir else base


@dataclass(frozen=True)
class ApplyConfig:
    enabled: bool
    target_hs: str
    input: ApplyInputConfig
    output: StagePathConfig
    write: ApplyWriteConfig
    products: dict[str, ApplyProductConfig]
    artifact: ApplyArtifactConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        scope: ScopeConfig,
    ) -> "ApplyConfig":
        target_hs = scope.target_hs

        # target_hs is governed by global.scope.target_hs.
        # If apply.target_hs is still present during migration, require equality.
        apply_target_hs = cfg.get("target_hs")
        if apply_target_hs is not None:
            parsed = require_str(
                apply_target_hs,
                key="apply.target_hs",
            ).strip().upper()
            if parsed != target_hs:
                raise ValueError(
                    "apply.target_hs must match global.scope.target_hs "
                    f"({target_hs!r}); got {parsed!r}."
                )

        context = TemplateContext(target_hs=target_hs)

        return cls(
            enabled=require_bool(
                cfg.get("enabled", False),
                key="apply.enabled",
            ),
            target_hs=target_hs,
            input=ApplyInputConfig.from_dict(
                require_dict(
                    cfg.get("input", {}),
                    key="apply.input",
                )
            ),
            output=StagePathConfig.from_dict(
                require_dict(
                    cfg.get("output", {}),
                    key="apply.output",
                ),
                key="apply.output",
                default_stage="interim",
                context=context,
            ),
            write=ApplyWriteConfig.from_dict(
                require_dict(
                    cfg.get("write", {}),
                    key="apply.write",
                )
            ),
            products={
                name: ApplyProductConfig.from_dict(name, item_cfg, context=context)
                for name, item_cfg in require_dict(
                    cfg.get("products", {}),
                    key="apply.products",
                ).items()
            },
            artifact=ApplyArtifactConfig.from_dict(
                require_dict(
                    cfg.get("artifact", {}),
                    key="apply.artifact",
                ),
                context=context,
            ),
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
    harvard_generator_audit: HarvardGeneratorAuditConfig
    harvard_mapping_direction: HarvardMappingDirectionConfig
    coverage: CoverageConfig
    weights: WeightsConfig
    apply: ApplyConfig

    references: dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> "HsSyncConfig":
        raw = load_hs_sync_config(config_path)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HsSyncConfig":
        global_cfg = _block(raw, "global")
        
        scope = ScopeConfig.from_dict(_block(global_cfg, "scope"))
        
        coverage = CoverageConfig.from_dict(
            _block(raw, "coverage"),
            scope=scope,
        )

        weights_cfg = require_dict(raw.get("weights", {}), key="weights")
        apply_cfg = require_dict(raw.get("apply", {}), key="apply")

        return cls(
            raw=raw,
            scope=scope,
            engine=EngineConfig.from_dict(_block(global_cfg, "engine", required=False)),
            progress=ProgressConfig.from_dict(_block(global_cfg, "progress", required=False)),
            columns=ColumnsConfig.from_dict(_block(global_cfg, "columns")),
            logging=LoggingConfig.from_dict(_block(global_cfg, "logging", required=False)),
            harvard_staging=HarvardStagingConfig.from_dict(
                _block(raw, "harvard_staging"),
                scope=scope,
            ),
            harvard_generator_audit=HarvardGeneratorAuditConfig.from_dict(
                _block(raw, "harvard_generator_audit", required=False),
                scope=scope,
            ),
            harvard_mapping_direction=HarvardMappingDirectionConfig.from_dict(
                _block(raw, "harvard_mapping_direction", required=False),
                scope=scope,
            ),
            coverage=coverage,
            weights=WeightsConfig.from_dict(
                weights_cfg,
                scope=scope,
            ),
            apply=ApplyConfig.from_dict(
                apply_cfg,
                scope=scope,
            ),
            references=_block(raw, "references", required=False),
        )
    @property
    def target_hs(self) -> str:
        """Effective target HS from global.scope.target_hs."""
        return self.scope.target_hs

    @property
    def template_context(self) -> TemplateContext:
        return TemplateContext(target_hs=self.target_hs)

    def render(self, value: str) -> str:
        return self.template_context.render(value)

    def product(self, name: str) -> ProductConfig:
        try:
            return self.coverage.products[name]
        except KeyError as exc:
            raise KeyError(f"Unknown coverage product: {name!r}") from exc

    def is_product_enabled(self, name: str, *, default: bool = False) -> bool:
        product = self.coverage.products.get(name)
        if product is None:
            return default
        return product.enabled

    def enabled_products(self) -> list[str]:
        return [
            product_name
            for product_name, product in self.coverage.products.items()
            if product.enabled
        ]

    def product_filename(self, name: str) -> str:
        return self.product(name).filename

    def product_options(self, name: str) -> dict[str, Any]:
        return self.product(name).options

    def product_write_enabled(self, name: str, *, default: bool = True) -> bool:
        product = self.coverage.products.get(name)
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

    def weights_output_dir(self) -> Path:
        return self.weights.output.resolve_data_dir()

    @staticmethod
    def _filename_with_ext(filename: str, ext: str) -> str:
        """
        Normalize a configured filename token.

        If filename already includes any suffix (e.g. '.csv', '.parquet'),
        return it unchanged. Otherwise append the requested extension.
        """
        name = require_str(filename, key="filename").strip()
        if Path(name).suffix:
            return name

        normalized_ext = ext.lstrip(".")
        return f"{name}.{normalized_ext}"

    def weights_product_path(self, name: str, *, ext: str = "csv") -> Path:
        product = self.weights.products.get(name)
        if product is None:
            raise KeyError(f"Unknown weights product: {name!r}")

        return self.weights_output_dir() / self._filename_with_ext(product.filename, ext)

    def weights_equal_split_weights_path(self) -> Path:
        if "equal_split_weights" not in self.weights.products:
            raise KeyError(
                "Missing required config: weights.products.equal_split_weights. "
                "Define this product in configs/hs_sync.yaml."
            )

        return self.weights_product_path("equal_split_weights")

    def weights_complete_mapping_weights_path(self) -> Path:
        if "complete_mapping_weights" not in self.weights.products:
            raise KeyError(
                "Missing required config: weights.products.complete_mapping_weights. "
                "Define this product in configs/hs_sync.yaml."
            )

        return self.weights_product_path("complete_mapping_weights")

    def weights_equal_split_audit_path(self) -> Path:
        return self.weights_artifact_output_dir() / self._filename_with_ext(
            self.weights.artifact.equal_split_audit_filename,
            "csv",
        )

    def weights_complete_mapping_audit_path(self) -> Path:
        return self.weights_artifact_output_dir() / self._filename_with_ext(
            self.weights.artifact.complete_weights_audit_filename,
            "csv",
        )

    def weights_artifact_output_dir(self) -> Path:
        return self.weights.artifact.resolve_output_dir()

    def weights_input_coverage_path(self) -> Path:
        filename = self.weights.products.get(
            self.weights.input.coverage_product
        )

        if filename is not None:
            coverage_filename = filename.filename
        else:
            coverage_filename = self.product_filename(
                self.weights.input.coverage_product
            )

        return self.coverage_output_dir() / self._filename_with_ext(
            coverage_filename,
            "csv",
        )

    def apply_output_dir(self) -> Path:
        return self.apply.output.resolve_data_dir()

    def apply_trade_data_dir(self) -> Path:
        ref = self.apply.input.trade_data.strip().lower()
        if ref == "coverage.input":
            return self.coverage_input_dir()

        raise ValueError(
            "Unsupported apply.input.trade_data reference: "
            f"{self.apply.input.trade_data!r}. Only 'coverage.input' is supported."
        )

    def apply_weights_path(self) -> Path:
        ref = self.apply.input.weights
        parts = [part.strip() for part in ref.split(".", 1)]
        if len(parts) != 2 or parts[0].lower() != "weights":
            raise ValueError(
                "apply.input.weights must be in the form 'weights.<product_name>', "
                f"got {ref!r}."
            )

        product_name = parts[1]
        product = self.weights.products.get(product_name)
        if product is None:
            raise KeyError(
                "Unknown weights product in apply.input.weights: "
                f"{product_name!r}. Define weights.products.{product_name} "
                "in configs/hs_sync.yaml."
            )

        filename = product.filename

        return self.weights_output_dir() / self._filename_with_ext(filename, "csv")

    def apply_output_path(self) -> Path:
        product = self.apply.products.get("hs_synced_trade")
        if product is None:
            raise KeyError(
                "Missing required config: apply.products.hs_synced_trade. "
                "Define this product in configs/hs_sync.yaml."
            )

        filename = product.filename

        return self.apply_output_dir() / self._filename_with_ext(
            filename,
            self.apply.write.format,
        )

    def apply_audit_path(self) -> Path:
        return self.apply_artifact_output_dir() / self._filename_with_ext(
            self.apply.artifact.audit_filename,
        "csv",
    )

    def apply_integrity_audit_path(self) -> Path:
        return self.apply_artifact_output_dir() / self._filename_with_ext(
            self.apply.artifact.integrity_audit_filename,
            "csv",
        )

    def apply_artifact_output_dir(self) -> Path:
        return self.apply.artifact.resolve_output_dir()

    def _resolve_reference_block_dir(
        self,
        *,
        block: dict[str, Any],
        key: str,) -> Path:
        source = require_str(
            block.get("source", "repo"),
            key=f"{key}.source",
        ).strip().lower()

        path = Path(
            require_str(
                block.get("path"),
                key=f"{key}.path",
            )
        )

        if source == "repo":
            # Resolve against repo.references; allow optional leading "references".
            if path.parts and path.parts[0] == "references":
                path = Path(*path.parts[1:])

            return get_repo_references_dir() / path

        if source == "data":
            # Resolve against data.references; allow optional leading "references".
            if path.parts and path.parts[0] == "references":
                path = Path(*path.parts[1:])

            return get_data_dir("references") / path

        raise ValueError(
            f"{key}.source must be 'repo' or 'data', got {source!r}."
        )

    def comtrade_hs_correlation_root(self) -> Path:
        try:
            block = self.references["comtrade"]["hs_correlation_groups"]
        except KeyError as exc:
            raise KeyError(
                "Missing required config block: "
                "references.comtrade.hs_correlation_groups"
            ) from exc

        return self._resolve_reference_block_dir(
            block=block,
            key="references.comtrade.hs_correlation_groups",
        )

    def comtrade_hs_correlation_raw_dir(self) -> Path:
        return self.comtrade_hs_correlation_root() / "raw"

    def comtrade_hs_correlation_normalized_dir(self) -> Path:
        return self.comtrade_hs_correlation_root() / "normalized"