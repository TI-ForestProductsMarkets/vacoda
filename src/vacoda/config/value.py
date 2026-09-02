"""
vacoda.config.value

Config reader for configs/steps/value.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vacoda.config.base import (
    get_step_config_path,
    load_yaml_config,
    require_bool,
    require_dict,
    require_str,
)
from vacoda.config.paths import (
    get_data_dir,
    get_output_dir,
    get_repo_references_dir,
)
from vacoda.config.pipeline import (
    PipelineEngineConfig,
    PipelineScopeConfig,
)


VALUE_CONFIG_PATH = get_step_config_path("value")


def load_value_config(
    config_path: Path | None = None,
) -> dict[str, Any]:
    return load_yaml_config(config_path or VALUE_CONFIG_PATH)


def _render_value(
    value: Any,
    *,
    target_hs: str,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _render_value(
                item,
                target_hs=target_hs,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _render_value(
                item,
                target_hs=target_hs,
            )
            for item in value
        ]

    if isinstance(value, str):
        return value.format(
            target_hs=target_hs.upper(),
            target_hs_upper=target_hs.upper(),
            target_hs_lower=target_hs.lower(),
        )

    return value


# ======================================================================
# Input / Output
# ======================================================================


@dataclass(frozen=True)
class ValueInputConfig:
    stage: str
    subdir: str
    filename: str
    format: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValueInputConfig":
        format_name = require_str(
            cfg.get("format"),
            key="value.input.format",
        ).strip().lower()

        if format_name != "parquet":
            raise ValueError(
                "value.input.format currently must be 'parquet', "
                f"got {format_name!r}."
            )

        return cls(
            stage=require_str(
                cfg.get("stage"),
                key="value.input.stage",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="value.input.subdir",
                allow_empty=True,
            ),
            filename=require_str(
                cfg.get("filename"),
                key="value.input.filename",
            ),
            format=format_name,
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.filename


@dataclass(frozen=True)
class ValueOutputConfig:
    stage: str
    subdir: str
    format: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValueOutputConfig":
        format_name = require_str(
            cfg.get("format"),
            key="value.output.format",
        ).strip().lower()

        if format_name != "parquet":
            raise ValueError(
                "value.output.format currently must be 'parquet', "
                f"got {format_name!r}."
            )

        return cls(
            stage=require_str(
                cfg.get("stage"),
                key="value.output.stage",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="value.output.subdir",
                allow_empty=True,
            ),
            format=format_name,
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base


# ======================================================================
# Reference data
# ======================================================================


@dataclass(frozen=True)
class ValueReferenceFileConfig:
    source: str
    path: str
    filename: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "ValueReferenceFileConfig":
        source = require_str(
            cfg.get("source"),
            key=f"{key}.source",
        ).strip().lower()

        if source != "repo":
            raise ValueError(
                f"{key}.source currently must be 'repo', "
                f"got {source!r}."
            )

        return cls(
            source=source,
            path=require_str(
                cfg.get("path"),
                key=f"{key}.path",
                allow_empty=True,
            ),
            filename=require_str(
                cfg.get("filename"),
                key=f"{key}.filename",
            ),
        )

    def resolve_dir(self) -> Path:
        base = get_repo_references_dir()
        return base / self.path if self.path else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.filename


@dataclass(frozen=True)
class ValueGDPDeflatorConfig:
    source: ValueReferenceFileConfig
    prepared: ValueReferenceFileConfig
    country_code: str
    indicator_code: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValueGDPDeflatorConfig":
        return cls(
            source=ValueReferenceFileConfig.from_dict(
                require_dict(
                    cfg.get("source"),
                    key="value.reference_data.us_gdp_deflator.source",
                ),
                key="value.reference_data.us_gdp_deflator.source",
            ),
            prepared=ValueReferenceFileConfig.from_dict(
                require_dict(
                    cfg.get("prepared"),
                    key="value.reference_data.us_gdp_deflator.prepared",
                ),
                key="value.reference_data.us_gdp_deflator.prepared",
            ),
            country_code=require_str(
                cfg.get("country_code"),
                key="value.reference_data.us_gdp_deflator.country_code",
            ).strip().upper(),
            indicator_code=require_str(
                cfg.get("indicator_code"),
                key="value.reference_data.us_gdp_deflator.indicator_code",
            ).strip(),
        )


@dataclass(frozen=True)
class ValueReferenceDataConfig:
    us_gdp_deflator: ValueGDPDeflatorConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValueReferenceDataConfig":
        return cls(
            us_gdp_deflator=ValueGDPDeflatorConfig.from_dict(
                require_dict(
                    cfg.get("us_gdp_deflator"),
                    key="value.reference_data.us_gdp_deflator",
                )
            )
        )


# ======================================================================
# Products
# ======================================================================


@dataclass(frozen=True)
class ValueProductConfig:
    enable: bool
    filename: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "ValueProductConfig":
        return cls(
            enable=require_bool(
                cfg.get("enable"),
                key=f"{key}.enable",
            ),
            filename=require_str(
                cfg.get("filename"),
                key=f"{key}.filename",
            ),
        )


@dataclass(frozen=True)
class ValueProductsConfig:
    deflated_trade: ValueProductConfig
    audit: ValueProductConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValueProductsConfig":

        def product(name: str) -> ValueProductConfig:
            key = f"value.products.{name}"

            return ValueProductConfig.from_dict(
                require_dict(
                    cfg.get(name),
                    key=key,
                ),
                key=key,
            )

        return cls(
            deflated_trade=product("deflated_trade"),
            audit=product("audit"),
        )


# ======================================================================
# Method
# ======================================================================


@dataclass(frozen=True)
class ValueMethodConfig:
    reference_year: int
    preserve_nominal_values: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValueMethodConfig":
        reference_year_raw = cfg.get("reference_year")

        if isinstance(reference_year_raw, bool):
            raise ValueError(
                "value.method.reference_year must be an integer year."
            )

        try:
            reference_year = int(reference_year_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "value.method.reference_year must be an integer year."
            ) from exc

        if reference_year < 1900:
            raise ValueError(
                "value.method.reference_year must be >= 1900."
            )

        return cls(
            reference_year=reference_year,
            preserve_nominal_values=require_bool(
                cfg.get("preserve_nominal_values"),
                key="value.method.preserve_nominal_values",
            ),
        )


# ======================================================================
# Logging
# ======================================================================


@dataclass(frozen=True)
class ValueLoggingConfig:
    level: str
    subdir: str
    file: str
    timestamp: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValueLoggingConfig":
        return cls(
            level=require_str(
                cfg.get("level"),
                key="value.logging.level",
            ).strip().upper(),
            subdir=require_str(
                cfg.get("subdir"),
                key="value.logging.subdir",
                allow_empty=True,
            ),
            file=require_str(
                cfg.get("file"),
                key="value.logging.file",
            ),
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="value.logging.timestamp",
            ),
        )

    def resolve_log_dir(self) -> Path:
        base = get_output_dir("logs")

        if not self.subdir:
            return base

        parts = list(Path(self.subdir).parts)

        if parts and parts[0].lower() == "logs":
            parts = parts[1:]

        return base / Path(*parts) if parts else base

    def resolve_log_file(self) -> Path:
        return self.resolve_log_dir() / self.file


# ======================================================================
# Artifacts
# ======================================================================


@dataclass(frozen=True)
class ValueArtifactsConfig:
    enable: bool
    kind: str
    subdir: str
    filename: str
    timestamp: bool
    write_latest: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "ValueArtifactsConfig":
        return cls(
            enable=require_bool(
                cfg.get("enable"),
                key="value.artifacts.enable",
            ),
            kind=require_str(
                cfg.get("kind"),
                key="value.artifacts.kind",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="value.artifacts.subdir",
                allow_empty=True,
            ),
            filename=require_str(
                cfg.get("filename"),
                key="value.artifacts.filename",
            ),
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="value.artifacts.timestamp",
            ),
            write_latest=require_bool(
                cfg.get("write_latest"),
                key="value.artifacts.write_latest",
            ),
        )

    def resolve_dir(self) -> Path:
        base = get_output_dir(self.kind)
        return base / self.subdir if self.subdir else base

    def timestamped_path(
        self,
        run_id: str,
    ) -> Path:
        filename = (
            f"{self.filename}_{run_id}.md"
            if self.timestamp
            else f"{self.filename}.md"
        )

        return self.resolve_dir() / filename

    def latest_path(self) -> Path:
        return (
            self.resolve_dir()
            / f"{self.filename}_latest.md"
        )


# ======================================================================
# Root config
# ======================================================================


@dataclass(frozen=True)
class ValueConfig:
    input: ValueInputConfig
    output: ValueOutputConfig
    reference_data: ValueReferenceDataConfig
    products: ValueProductsConfig
    method: ValueMethodConfig
    logging: ValueLoggingConfig
    artifacts: ValueArtifactsConfig
    scope: PipelineScopeConfig
    engine: PipelineEngineConfig

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | None = None,
        *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "ValueConfig":
        return cls.from_dict(
            load_value_config(config_path),
            scope=scope,
            engine=engine,
        )

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "ValueConfig":
        if scope is None:
            raise ValueError(
                "ValueConfig requires pipeline scope."
            )

        if engine is None:
            raise ValueError(
                "ValueConfig requires pipeline engine."
            )

        root = require_dict(
            raw.get("value"),
            key="value",
        )

        rendered = require_dict(
            _render_value(
                root,
                target_hs=scope.target_hs,
            ),
            key="value",
        )

        return cls(
            input=ValueInputConfig.from_dict(
                require_dict(
                    rendered.get("input"),
                    key="value.input",
                )
            ),
            output=ValueOutputConfig.from_dict(
                require_dict(
                    rendered.get("output"),
                    key="value.output",
                )
            ),
            reference_data=ValueReferenceDataConfig.from_dict(
                require_dict(
                    rendered.get("reference_data"),
                    key="value.reference_data",
                )
            ),
            products=ValueProductsConfig.from_dict(
                require_dict(
                    rendered.get("products"),
                    key="value.products",
                )
            ),
            method=ValueMethodConfig.from_dict(
                require_dict(
                    rendered.get("method"),
                    key="value.method",
                )
            ),
            logging=ValueLoggingConfig.from_dict(
                require_dict(
                    rendered.get("logging"),
                    key="value.logging",
                )
            ),
            artifacts=ValueArtifactsConfig.from_dict(
                require_dict(
                    rendered.get("artifacts"),
                    key="value.artifacts",
                )
            ),
            scope=scope,
            engine=engine,
        )

    # ------------------------------------------------------------------
    # Resolved paths
    # ------------------------------------------------------------------

    def input_path(self) -> Path:
        return self.input.resolve_path()

    def output_dir(self) -> Path:
        return self.output.resolve_dir()

    def deflator_source_path(self) -> Path:
        return (
            self.reference_data
            .us_gdp_deflator
            .source
            .resolve_path()
        )

    def deflator_prepared_path(self) -> Path:
        return (
            self.reference_data
            .us_gdp_deflator
            .prepared
            .resolve_path()
        )

    def product_path(
        self,
        product_name: str,
    ) -> Path:
        products = {
            "deflated_trade": self.products.deflated_trade,
            "audit": self.products.audit,
        }

        if product_name not in products:
            raise KeyError(
                f"Unknown Value product: {product_name!r}"
            )

        return (
            self.output_dir()
            / products[product_name].filename
        )

    def resolved_product_paths(self) -> dict[str, Path]:
        return {
            name: self.product_path(name)
            for name in (
                "deflated_trade",
                "audit",
            )
        }

    def enabled_products(self) -> dict[str, bool]:
        return {
            "deflated_trade": (
                self.products.deflated_trade.enable
            ),
            "audit": self.products.audit.enable,
        }

    def log_path(self) -> Path:
        return self.logging.resolve_log_file()

    def artifacts_path(
        self,
        run_id: str,
    ) -> Path:
        return self.artifacts.timestamped_path(run_id)

    def artifacts_latest_path(self) -> Path:
        return self.artifacts.latest_path()