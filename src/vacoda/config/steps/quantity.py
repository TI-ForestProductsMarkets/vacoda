"""
vacoda.config.quantity

Config reader for configs/steps/quantity.yaml.
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


QUANTITY_CONFIG_PATH = get_step_config_path("quantity")


def load_quantity_config(
    config_path: Path | None = None,
) -> dict[str, Any]:
    return load_yaml_config(config_path or QUANTITY_CONFIG_PATH)


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


# ======================================================================
# Input / Output
# ======================================================================


@dataclass(frozen=True)
class QuantityInputConfig:
    stage: str
    subdir: str
    filename: str
    format: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityInputConfig":
        format_name = require_str(
            cfg.get("format"),
            key="quantity.input.format",
        ).strip().lower()

        if format_name != "parquet":
            raise ValueError(
                "quantity.input.format currently must be 'parquet', "
                f"got {format_name!r}."
            )

        return cls(
            stage=require_str(
                cfg.get("stage"),
                key="quantity.input.stage",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="quantity.input.subdir",
                allow_empty=True,
            ),
            filename=require_str(
                cfg.get("filename"),
                key="quantity.input.filename",
            ),
            format=format_name,
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.filename


@dataclass(frozen=True)
class QuantityOutputConfig:
    stage: str
    subdir: str
    format: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityOutputConfig":
        format_name = require_str(
            cfg.get("format"),
            key="quantity.output.format",
        ).strip().lower()

        if format_name != "parquet":
            raise ValueError(
                "quantity.output.format currently must be 'parquet', "
                f"got {format_name!r}."
            )

        return cls(
            stage=require_str(
                cfg.get("stage"),
                key="quantity.output.stage",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="quantity.output.subdir",
                allow_empty=True,
            ),
            format=format_name,
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base


# ======================================================================
# Products
# ======================================================================


@dataclass(frozen=True)
class QuantityProductConfig:
    enable: bool
    filename: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "QuantityProductConfig":
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
class QuantityProductsConfig:
    prepared_input: QuantityProductConfig
    reconciliation_evidence: QuantityProductConfig
    synchronized_quantity: QuantityProductConfig
    diagnostics: QuantityProductConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityProductsConfig":

        def product(name: str) -> QuantityProductConfig:
            key = f"quantity.products.{name}"

            return QuantityProductConfig.from_dict(
                require_dict(
                    cfg.get(name),
                    key=key,
                ),
                key=key,
            )

        return cls(
            prepared_input=product("prepared_input"),
            reconciliation_evidence=product(
                "reconciliation_evidence"
            ),
            synchronized_quantity=product(
                "synchronized_quantity"
            ),
            diagnostics=product("diagnostics"),
        )


# ======================================================================
# Method
# ======================================================================


@dataclass(frozen=True)
class QuantitySourceSelectionConfig:
    method: Literal["data_driven"]

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantitySourceSelectionConfig":
        method = require_str(
            cfg.get("method"),
            key=(
                "quantity.method.kg_reconciliation."
                "source_selection.method"
            ),
        ).strip().lower()

        allowed = {"data_driven"}

        if method not in allowed:
            raise ValueError(
                "quantity.method.kg_reconciliation."
                "source_selection.method must be one of "
                f"{sorted(allowed)}, got {method!r}."
            )

        return cls(method=method)


@dataclass(frozen=True)
class QuantityAgreementConfig:
    relative_tolerance: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityAgreementConfig":
        return cls(
            relative_tolerance=require_float(
                cfg.get("relative_tolerance"),
                key=(
                    "quantity.method.kg_reconciliation."
                    "agreement.relative_tolerance"
                ),
                min_value=0.0,
                max_value=1.0,
            )
        )


@dataclass(frozen=True)
class QuantityKgReconciliationConfig:
    require_positive_values: bool
    quantity_unit: str
    source_selection: QuantitySourceSelectionConfig
    agreement: QuantityAgreementConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityKgReconciliationConfig":
        quantity_unit = require_str(
            cfg.get("quantity_unit"),
            key=(
                "quantity.method.kg_reconciliation."
                "quantity_unit"
            ),
        ).strip().lower()

        if not quantity_unit:
            raise ValueError(
                "quantity.method.kg_reconciliation."
                "quantity_unit must not be empty."
            )

        return cls(
            require_positive_values=require_bool(
                cfg.get("require_positive_values"),
                key=(
                    "quantity.method.kg_reconciliation."
                    "require_positive_values"
                ),
            ),
            quantity_unit=quantity_unit,
            source_selection=(
                QuantitySourceSelectionConfig.from_dict(
                    require_dict(
                        cfg.get("source_selection"),
                        key=(
                            "quantity.method.kg_reconciliation."
                            "source_selection"
                        ),
                    )
                )
            ),
            agreement=QuantityAgreementConfig.from_dict(
                require_dict(
                    cfg.get("agreement"),
                    key=(
                        "quantity.method.kg_reconciliation."
                        "agreement"
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class QuantityGrossWeightConfig:
    use_as_quantity_proxy: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityGrossWeightConfig":
        return cls(
            use_as_quantity_proxy=require_bool(
                cfg.get("use_as_quantity_proxy"),
                key=(
                    "quantity.method.gross_weight."
                    "use_as_quantity_proxy"
                ),
            )
        )


@dataclass(frozen=True)
class QuantityMethodConfig:
    kg_reconciliation: QuantityKgReconciliationConfig
    gross_weight: QuantityGrossWeightConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityMethodConfig":
        return cls(
            kg_reconciliation=(
                QuantityKgReconciliationConfig.from_dict(
                    require_dict(
                        cfg.get("kg_reconciliation"),
                        key="quantity.method.kg_reconciliation",
                    )
                )
            ),
            gross_weight=QuantityGrossWeightConfig.from_dict(
                require_dict(
                    cfg.get("gross_weight"),
                    key="quantity.method.gross_weight",
                )
            ),
        )


# ======================================================================
# Diagnostics
# ======================================================================


@dataclass(frozen=True)
class QuantityDiagnosticToggleConfig:
    enabled: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "QuantityDiagnosticToggleConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key=f"{key}.enabled",
            )
        )


@dataclass(frozen=True)
class QuantityNetWeightComparisonConfig:
    enabled: bool
    quantity_unit: str

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityNetWeightComparisonConfig":
        return cls(
            enabled=require_bool(
                cfg.get("enabled"),
                key=(
                    "quantity.diagnostics."
                    "quantity_net_weight_comparison.enabled"
                ),
            ),
            quantity_unit=require_str(
                cfg.get("quantity_unit"),
                key=(
                    "quantity.diagnostics."
                    "quantity_net_weight_comparison.quantity_unit"
                ),
            ).strip().lower(),
        )


@dataclass(frozen=True)
class QuantityDiagnosticsConfig:
    source_coverage: QuantityDiagnosticToggleConfig
    reported_estimated_overlap: QuantityDiagnosticToggleConfig
    quantity_net_weight_comparison: (
        QuantityNetWeightComparisonConfig
    )
    unit_distribution: QuantityDiagnosticToggleConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityDiagnosticsConfig":

        def toggle(name: str) -> QuantityDiagnosticToggleConfig:
            key = f"quantity.diagnostics.{name}"

            return QuantityDiagnosticToggleConfig.from_dict(
                require_dict(
                    cfg.get(name),
                    key=key,
                ),
                key=key,
            )

        return cls(
            source_coverage=toggle("source_coverage"),
            reported_estimated_overlap=toggle(
                "reported_estimated_overlap"
            ),
            quantity_net_weight_comparison=(
                QuantityNetWeightComparisonConfig.from_dict(
                    require_dict(
                        cfg.get(
                            "quantity_net_weight_comparison"
                        ),
                        key=(
                            "quantity.diagnostics."
                            "quantity_net_weight_comparison"
                        ),
                    )
                )
            ),
            unit_distribution=toggle("unit_distribution"),
        )


# ======================================================================
# Logging
# ======================================================================


@dataclass(frozen=True)
class QuantityLoggingConfig:
    level: str
    subdir: str
    file: str
    timestamp: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "QuantityLoggingConfig":
        return cls(
            level=require_str(
                cfg.get("level"),
                key="quantity.logging.level",
            ).strip().upper(),
            subdir=require_str(
                cfg.get("subdir"),
                key="quantity.logging.subdir",
                allow_empty=True,
            ),
            file=require_str(
                cfg.get("file"),
                key="quantity.logging.file",
            ),
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="quantity.logging.timestamp",
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
class QuantityArtifactsConfig:
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
    ) -> "QuantityArtifactsConfig":
        return cls(
            enable=require_bool(
                cfg.get("enable"),
                key="quantity.artifacts.enable",
            ),
            kind=require_str(
                cfg.get("kind"),
                key="quantity.artifacts.kind",
            ).strip().lower(),
            subdir=require_str(
                cfg.get("subdir"),
                key="quantity.artifacts.subdir",
                allow_empty=True,
            ),
            filename=require_str(
                cfg.get("filename"),
                key="quantity.artifacts.filename",
            ),
            timestamp=require_bool(
                cfg.get("timestamp"),
                key="quantity.artifacts.timestamp",
            ),
            write_latest=require_bool(
                cfg.get("write_latest"),
                key="quantity.artifacts.write_latest",
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


# ======================================================================
# Root config
# ======================================================================


@dataclass(frozen=True)
class QuantityConfig:
    input: QuantityInputConfig
    output: QuantityOutputConfig
    products: QuantityProductsConfig
    method: QuantityMethodConfig
    diagnostics: QuantityDiagnosticsConfig
    logging: QuantityLoggingConfig
    artifacts: QuantityArtifactsConfig
    scope: PipelineScopeConfig
    engine: PipelineEngineConfig

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | None = None,
        *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "QuantityConfig":
        return cls.from_dict(
            load_quantity_config(config_path),
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
    ) -> "QuantityConfig":
        if scope is None:
            raise ValueError(
                "QuantityConfig requires pipeline scope."
            )

        if engine is None:
            raise ValueError(
                "QuantityConfig requires pipeline engine."
            )

        root = require_dict(
            raw.get("quantity"),
            key="quantity",
        )

        rendered = require_dict(
            _render_value(
                root,
                target_hs=scope.target_hs,
            ),
            key="quantity",
        )

        return cls(
            input=QuantityInputConfig.from_dict(
                require_dict(
                    rendered.get("input"),
                    key="quantity.input",
                )
            ),
            output=QuantityOutputConfig.from_dict(
                require_dict(
                    rendered.get("output"),
                    key="quantity.output",
                )
            ),
            products=QuantityProductsConfig.from_dict(
                require_dict(
                    rendered.get("products"),
                    key="quantity.products",
                )
            ),
            method=QuantityMethodConfig.from_dict(
                require_dict(
                    rendered.get("method"),
                    key="quantity.method",
                )
            ),
            diagnostics=QuantityDiagnosticsConfig.from_dict(
                require_dict(
                    rendered.get("diagnostics"),
                    key="quantity.diagnostics",
                )
            ),
            logging=QuantityLoggingConfig.from_dict(
                require_dict(
                    rendered.get("logging"),
                    key="quantity.logging",
                )
            ),
            artifacts=QuantityArtifactsConfig.from_dict(
                require_dict(
                    rendered.get("artifacts"),
                    key="quantity.artifacts",
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

    def product_path(self, product_name: str) -> Path:
        products = {
            "prepared_input": self.products.prepared_input,
            "reconciliation_evidence": (
                self.products.reconciliation_evidence
            ),
            "synchronized_quantity": (
                self.products.synchronized_quantity
            ),
            "diagnostics": self.products.diagnostics,
        }

        if product_name not in products:
            raise KeyError(
                f"Unknown Quantity product: {product_name!r}"
            )

        return (
            self.output_dir()
            / products[product_name].filename
        )

    def resolved_product_paths(self) -> dict[str, Path]:
        return {
            name: self.product_path(name)
            for name in (
                "prepared_input",
                "reconciliation_evidence",
                "synchronized_quantity",
                "diagnostics",
            )
        }

    def enabled_products(self) -> dict[str, bool]:
        return {
            "prepared_input": (
                self.products.prepared_input.enable
            ),
            "reconciliation_evidence": (
                self.products.reconciliation_evidence.enable
            ),
            "synchronized_quantity": (
                self.products.synchronized_quantity.enable
            ),
            "diagnostics": self.products.diagnostics.enable,
        }

    def log_path(self) -> Path:
        return self.logging.resolve_log_file()

    def artifacts_path(self, run_id: str) -> Path:
        return self.artifacts.timestamped_path(run_id)

    def artifacts_latest_path(self) -> Path:
        return self.artifacts.latest_path()