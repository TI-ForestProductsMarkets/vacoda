"""
vacoda.config.trade_sync

Config reader for configs/steps/trade_sync.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vacoda.config.base import (
    get_step_config_path,
    load_yaml_config,
    require_bool,
    require_dict,
    require_float,
    require_str,
)
from vacoda.config.paths import get_data_dir, get_output_dir
from vacoda.config.pipeline import (
    PipelineEngineConfig,
    PipelineScopeConfig,
)

TRADE_SYNC_CONFIG_PATH = get_step_config_path("trade_sync")


def load_trade_sync_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_yaml_config(config_path or TRADE_SYNC_CONFIG_PATH)


def _render_value(value: Any, *, target_hs: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _render_value(item, target_hs=target_hs)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _render_value(item, target_hs=target_hs)
            for item in value
        ]
    if isinstance(value, str):
        return value.format(
            target_hs=target_hs.upper(),
            target_hs_upper=target_hs.upper(),
            target_hs_lower=target_hs.lower(),
        )
    return value


def _require_float_list(
    value: Any,
    *,
    key: str,
    min_value: float | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list.")

    result = tuple(
        require_float(item, key=key, min_value=min_value)
        for item in value
    )

    if tuple(sorted(result)) != result:
        raise ValueError(f"{key} must be sorted in ascending order.")

    if len(set(result)) != len(result):
        raise ValueError(f"{key} must not contain duplicate values.")

    return result


def _require_str_list(value: Any, *, key: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list.")

    result = tuple(
        require_str(item, key=key).strip()
        for item in value
    )

    if any(not item for item in result):
        raise ValueError(f"{key} must not contain empty values.")

    if len(set(result)) != len(result):
        raise ValueError(f"{key} must not contain duplicate values.")

    return result


@dataclass(frozen=True)
class TradeSyncInputConfig:
    stage: str
    subdir: str
    filename: str
    format: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TradeSyncInputConfig":
        format_name = require_str(
            cfg.get("format"),
            key="trade_sync.input.format",
        ).strip().lower()

        if format_name != "parquet":
            raise ValueError(
                "trade_sync.input.format currently must be 'parquet', "
                f"got {format_name!r}."
            )

        return cls(
            stage=require_str(
                cfg.get("stage"),
                key="trade_sync.input.stage",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="trade_sync.input.subdir",
                allow_empty=True,
            ),
            filename=require_str(
                cfg.get("filename"),
                key="trade_sync.input.filename",
            ),
            format=format_name,
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.filename


@dataclass(frozen=True)
class TradeSyncOutputConfig:
    stage: str
    subdir: str
    format: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TradeSyncOutputConfig":
        format_name = require_str(
            cfg.get("format"),
            key="trade_sync.output.format",
        ).strip().lower()

        if format_name != "parquet":
            raise ValueError(
                "trade_sync.output.format currently must be 'parquet', "
                f"got {format_name!r}."
            )

        return cls(
            stage=require_str(
                cfg.get("stage"),
                key="trade_sync.output.stage",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="trade_sync.output.subdir",
                allow_empty=True,
            ),
            format=format_name,
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base


@dataclass(frozen=True)
class TradeSyncLoggingConfig:
    level: str
    subdir: str
    file: str
    timestamp: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TradeSyncLoggingConfig":
        return cls(
            level=require_str(
                cfg.get("level"),
                key="trade_sync.logging.level",
            ).strip().upper(),
            subdir=require_str(
                cfg.get("subdir"),
                key="trade_sync.logging.subdir",
                allow_empty=True,
            ),
            file=require_str(
                cfg.get("file"),
                key="trade_sync.logging.file",
            ),
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="trade_sync.logging.timestamp",
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


@dataclass(frozen=True)
class TradeSyncProductConfig:
    enable: bool
    filename: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "TradeSyncProductConfig":
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
class TradeSyncProductsConfig:
    prepared_input: TradeSyncProductConfig
    bilateral_pairs: TradeSyncProductConfig
    reliability_evidence: TradeSyncProductConfig
    country_reliability: TradeSyncProductConfig
    reconciled_trade: TradeSyncProductConfig
    reconciliation_audit: TradeSyncProductConfig
    diagnostics: TradeSyncProductConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TradeSyncProductsConfig":
        def product(name: str) -> TradeSyncProductConfig:
            key = f"trade_sync.products.{name}"
            return TradeSyncProductConfig.from_dict(
                require_dict(cfg.get(name), key=key),
                key=key,
            )

        return cls(
            prepared_input=product("prepared_input"),
            bilateral_pairs=product("bilateral_pairs"),
            reliability_evidence=product("reliability_evidence"),
            country_reliability=product("country_reliability"),
            reconciled_trade=product("reconciled_trade"),
            reconciliation_audit=product("reconciliation_audit"),
            diagnostics=product("diagnostics"),
        )


@dataclass(frozen=True)
class TradeSyncAgreementToleranceConfig:
    enabled: bool
    relative_difference: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "TradeSyncAgreementToleranceConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key=(
                    "trade_sync.method.reconciliation."
                    "agreement_tolerance.enabled"
                ),
            ),
            relative_difference=require_float(
                cfg.get("relative_difference"),
                key=(
                    "trade_sync.method.reconciliation."
                    "agreement_tolerance.relative_difference"
                ),
                min_value=0.0,
                max_value=1.0,
            ),
        )


@dataclass(frozen=True)
class TradeSyncSourceWeightsConfig:
    exporter_reported: float
    importer_reported: float
    exporter_estimated: float
    importer_estimated: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "TradeSyncSourceWeightsConfig":
        result = cls(
            exporter_reported=require_float(
                cfg.get("exporter_reported"),
                key=(
                    "trade_sync.method.reconciliation."
                    "source_weights.exporter_reported"
                ),
                min_value=0.0,
            ),
            importer_reported=require_float(
                cfg.get("importer_reported"),
                key=(
                    "trade_sync.method.reconciliation."
                    "source_weights.importer_reported"
                ),
                min_value=0.0,
            ),
            exporter_estimated=require_float(
                cfg.get("exporter_estimated"),
                key=(
                    "trade_sync.method.reconciliation."
                    "source_weights.exporter_estimated"
                ),
                min_value=0.0,
            ),
            importer_estimated=require_float(
                cfg.get("importer_estimated"),
                key=(
                    "trade_sync.method.reconciliation."
                    "source_weights.importer_estimated"
                ),
                min_value=0.0,
            ),
        )

        if max(
            result.exporter_reported,
            result.importer_reported,
            result.exporter_estimated,
            result.importer_estimated,
        ) <= 0:
            raise ValueError(
                "At least one Trade Sync source weight must be positive."
            )

        return result


@dataclass(frozen=True)
class TradeSyncBoundsConfig:
    minimum_reconciled_value: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TradeSyncBoundsConfig":
        return cls(
            minimum_reconciled_value=require_float(
                cfg.get("minimum_reconciled_value"),
                key=(
                    "trade_sync.method.reconciliation."
                    "bounds.minimum_reconciled_value"
                ),
                min_value=0.0,
            )
        )


@dataclass(frozen=True)
class TradeSyncReconciliationConfig:
    method: Literal[
        "weighted_mean",
        "geometric_mean",
        "reliability_weighted",
    ]
    require_positive_values: bool
    retain_unmatched_flows: bool
    source_weights: TradeSyncSourceWeightsConfig
    bounds: TradeSyncBoundsConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "TradeSyncReconciliationConfig":
        method = require_str(
            cfg.get("method"),
            key="trade_sync.method.reconciliation.method",
        ).strip().lower()

        allowed = {
            "weighted_mean",
            "geometric_mean",
            "reliability_weighted",
        }

        if method not in allowed:
            raise ValueError(
                "trade_sync.method.reconciliation.method must be one "
                f"of {sorted(allowed)}, got {method!r}."
            )

        return cls(
            method=method,
            require_positive_values=require_bool(
                cfg.get("require_positive_values"),
                key=(
                    "trade_sync.method.reconciliation."
                    "require_positive_values"
                ),
            ),
            retain_unmatched_flows=require_bool(
                cfg.get("retain_unmatched_flows"),
                key=(
                    "trade_sync.method.reconciliation."
                    "retain_unmatched_flows"
                ),
            ),
            source_weights=TradeSyncSourceWeightsConfig.from_dict(
                require_dict(
                    cfg.get("source_weights"),
                    key=(
                        "trade_sync.method.reconciliation."
                        "source_weights"
                    ),
                )
            ),
            bounds=TradeSyncBoundsConfig.from_dict(
                require_dict(
                    cfg.get("bounds"),
                    key="trade_sync.method.reconciliation.bounds",
                )
            ),
        )


@dataclass(frozen=True)
class TradeSyncDuplicateKeysValidationConfig:
    fail_on_duplicates: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "TradeSyncDuplicateKeysValidationConfig":
        return cls(
            fail_on_duplicates=require_bool(
                cfg.get("fail_on_duplicates"),
                key=(
                    "trade_sync.method.validation."
                    "duplicate_keys.fail_on_duplicates"
                ),
            )
        )


@dataclass(frozen=True)
class TradeSyncValuePreservationValidationConfig:
    enabled: bool
    relative_tolerance: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "TradeSyncValuePreservationValidationConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key=(
                    "trade_sync.method.validation."
                    "value_preservation.enabled"
                ),
            ),
            relative_tolerance=require_float(
                cfg.get("relative_tolerance"),
                key=(
                    "trade_sync.method.validation."
                    "value_preservation.relative_tolerance"
                ),
                min_value=0.0,
            ),
        )


@dataclass(frozen=True)
class TradeSyncBilateralConsistencyValidationConfig:
    enabled: bool
    maximum_relative_gap: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "TradeSyncBilateralConsistencyValidationConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key=(
                    "trade_sync.method.validation."
                    "bilateral_consistency.enabled"
                ),
            ),
            maximum_relative_gap=require_float(
                cfg.get("maximum_relative_gap"),
                key=(
                    "trade_sync.method.validation."
                    "bilateral_consistency.maximum_relative_gap"
                ),
                min_value=0.0,
                max_value=1.0,
            ),
        )


@dataclass(frozen=True)
class TradeSyncValidationConfig:
    duplicate_keys: TradeSyncDuplicateKeysValidationConfig
    value_preservation: TradeSyncValuePreservationValidationConfig
    bilateral_consistency: TradeSyncBilateralConsistencyValidationConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TradeSyncValidationConfig":
        return cls(
            duplicate_keys=(
                TradeSyncDuplicateKeysValidationConfig.from_dict(
                    require_dict(
                        cfg.get("duplicate_keys"),
                        key=(
                            "trade_sync.method.validation."
                            "duplicate_keys"
                        ),
                    )
                )
            ),
            value_preservation=(
                TradeSyncValuePreservationValidationConfig.from_dict(
                    require_dict(
                        cfg.get("value_preservation"),
                        key=(
                            "trade_sync.method.validation."
                            "value_preservation"
                        ),
                    )
                )
            ),
            bilateral_consistency=(
                TradeSyncBilateralConsistencyValidationConfig.from_dict(
                    require_dict(
                        cfg.get("bilateral_consistency"),
                        key=(
                            "trade_sync.method.validation."
                            "bilateral_consistency"
                        ),
                    )
                )
            ),
        )


@dataclass(frozen=True)
class TradeSyncMethodConfig:
    reconciliation: TradeSyncReconciliationConfig
    validation: TradeSyncValidationConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TradeSyncMethodConfig":
        return cls(
            reconciliation=TradeSyncReconciliationConfig.from_dict(
                require_dict(
                    cfg.get("reconciliation"),
                    key="trade_sync.method.reconciliation",
                )
            ),
            validation=TradeSyncValidationConfig.from_dict(
                require_dict(
                    cfg.get("validation"),
                    key="trade_sync.method.validation",
                )
            ),
        )


@dataclass(frozen=True)
class TradeSyncDiagnosticsConfig:
    value_bands_usd: tuple[float, ...]
    agreement_tolerance: TradeSyncAgreementToleranceConfig
    reconciliation_statuses: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "TradeSyncDiagnosticsConfig":
        statuses = _require_str_list(
            cfg.get("reconciliation_statuses"),
            key=(
                "trade_sync.diagnostics."
                "reconciliation_statuses"
            ),
        )

        allowed = {
            "bilateral_reconciled",
            "import_only",
            "export_only",
            "already_aligned",
            "not_eligible",
        }

        unknown = sorted(set(statuses) - allowed)
        if unknown:
            raise ValueError(
                "Unknown trade_sync reconciliation statuses: "
                + ", ".join(unknown)
            )

        return cls(
            value_bands_usd=_require_float_list(
                cfg.get("value_bands_usd"),
                key="trade_sync.diagnostics.value_bands_usd",
                min_value=0.0,
            ),
            agreement_tolerance=TradeSyncAgreementToleranceConfig.from_dict(
                require_dict(
                    cfg.get("agreement_tolerance"),
                    key="trade_sync.diagnostics.agreement_tolerance",
                )
            ),
            reconciliation_statuses=statuses,
        )


@dataclass(frozen=True)
class TradeSyncArtifactsConfig:
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
    ) -> "TradeSyncArtifactsConfig":
        return cls(
            enable=require_bool(
                cfg.get("enable"),
                key="trade_sync.artifacts.enable",
            ),
            kind=require_str(
                cfg.get("kind"),
                key="trade_sync.artifacts.kind",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="trade_sync.artifacts.subdir",
                allow_empty=True,
            ),
            filename=require_str(
                cfg.get("filename"),
                key="trade_sync.artifacts.filename",
            ),
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="trade_sync.artifacts.timestamp",
            ),
            write_latest=require_bool(
                cfg.get("write_latest"),
                key="trade_sync.artifacts.write_latest",
            ),
        )

    def resolve_dir(self) -> Path:
        base = get_output_dir(self.kind)
        return base / self.subdir if self.subdir else base

    def timestamped_path(self, run_id: str) -> Path:
        filename = (
            f"{self.filename}_{run_id}.md"
            if self.timestamp
            else f"{self.filename}.md"
        )
        return self.resolve_dir() / filename

    def latest_path(self) -> Path:
        return self.resolve_dir() / f"{self.filename}_latest.md"


@dataclass(frozen=True)
class TradeSyncConfig:
    input: TradeSyncInputConfig
    output: TradeSyncOutputConfig
    products: TradeSyncProductsConfig
    method: TradeSyncMethodConfig
    diagnostics: TradeSyncDiagnosticsConfig
    logging: TradeSyncLoggingConfig
    artifacts: TradeSyncArtifactsConfig
    scope: PipelineScopeConfig
    engine: PipelineEngineConfig

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | None = None,
        *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "TradeSyncConfig":
        return cls.from_dict(
            load_trade_sync_config(config_path),
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
    ) -> "TradeSyncConfig":
        if scope is None:
            raise ValueError(
                "TradeSyncConfig requires pipeline scope."
            )
        if engine is None:
            raise ValueError(
                "TradeSyncConfig requires pipeline engine."
            )

        root = require_dict(
            raw.get("trade_sync"),
            key="trade_sync",
        )

        rendered = require_dict(
            _render_value(root, target_hs=scope.target_hs),
            key="trade_sync",
        )

        return cls(
            input=TradeSyncInputConfig.from_dict(
                require_dict(
                    rendered.get("input"),
                    key="trade_sync.input",
                )
            ),
            output=TradeSyncOutputConfig.from_dict(
                require_dict(
                    rendered.get("output"),
                    key="trade_sync.output",
                )
            ),
            products=TradeSyncProductsConfig.from_dict(
                require_dict(
                    rendered.get("products"),
                    key="trade_sync.products",
                )
            ),
            method=TradeSyncMethodConfig.from_dict(
                require_dict(
                    rendered.get("method"),
                    key="trade_sync.method",
                )
            ),
            diagnostics=TradeSyncDiagnosticsConfig.from_dict(
                require_dict(
                    rendered.get("diagnostics"),
                    key="trade_sync.diagnostics",
                )
            ),
            logging=TradeSyncLoggingConfig.from_dict(
                require_dict(
                    rendered.get("logging"),
                    key="trade_sync.logging",
                )
            ),
            artifacts=TradeSyncArtifactsConfig.from_dict(
                require_dict(
                    rendered.get("artifacts"),
                    key="trade_sync.artifacts",
                )
            ),
            scope=scope,
            engine=engine,
        )

    def input_path(self) -> Path:
        return self.input.resolve_path()

    def output_dir(self) -> Path:
        return self.output.resolve_dir()

    def product_path(self, product_name: str) -> Path:
        products = {
            "prepared_input": self.products.prepared_input,
            "bilateral_pairs": self.products.bilateral_pairs,
            "reliability_evidence": self.products.reliability_evidence,
            "country_reliability": self.products.country_reliability,
            "reconciled_trade": self.products.reconciled_trade,
            "reconciliation_audit": self.products.reconciliation_audit,
            "diagnostics": self.products.diagnostics,
        }

        if product_name not in products:
            raise KeyError(
                f"Unknown Trade Sync product: {product_name!r}"
            )

        return self.output_dir() / products[product_name].filename

    def resolved_product_paths(self) -> dict[str, Path]:
        return {
            name: self.product_path(name)
            for name in (
                "prepared_input",
                "bilateral_pairs",
                "reliability_evidence",
                "country_reliability",
                "reconciled_trade",
                "reconciliation_audit",
                "diagnostics",
            )
        }

    def enabled_products(self) -> dict[str, bool]:
        return {
            "prepared_input": self.products.prepared_input.enable,
            "bilateral_pairs": self.products.bilateral_pairs.enable,
            "reliability_evidence": self.products.reliability_evidence.enable,
            "country_reliability": self.products.country_reliability.enable,
            "reconciled_trade": self.products.reconciled_trade.enable,
            "reconciliation_audit": self.products.reconciliation_audit.enable,
            "diagnostics": self.products.diagnostics.enable,
        }

    def log_path(self) -> Path:
        return self.logging.resolve_log_file()

    def artifacts_path(self, run_id: str) -> Path:
        return self.artifacts.timestamped_path(run_id)

    def artifacts_latest_path(self) -> Path:
        return self.artifacts.latest_path()