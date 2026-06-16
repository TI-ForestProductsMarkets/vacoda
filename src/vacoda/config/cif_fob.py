"""
vacoda.config.cif_fob

Config reader for configs/steps/cif_fob.yaml.
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
    require_float,
    require_str,
)
from vacoda.config.paths import get_data_dir
from vacoda.config.pipeline import PipelineEngineConfig, PipelineScopeConfig


CIF_FOB_CONFIG_PATH = get_step_config_path("cif_fob")


def load_cif_fob_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_yaml_config(config_path or CIF_FOB_CONFIG_PATH)


def _render_value(value: Any, *, target_hs: str) -> Any:
    if isinstance(value, dict):
        return {k: _render_value(v, target_hs=target_hs) for k, v in value.items()}

    if isinstance(value, list):
        return [_render_value(item, target_hs=target_hs) for item in value]

    if isinstance(value, str):
        return value.format(
            target_hs=target_hs.upper(),
            target_hs_upper=target_hs.upper(),
            target_hs_lower=target_hs.lower(),
        )

    return value


@dataclass(frozen=True)
class CifFobInputConfig:
    stage: str
    subdir: str
    filename: str
    format: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobInputConfig":
        return cls(
            stage=require_str(cfg.get("stage"), key="cif_fob.input.stage").strip().lower(),
            subdir=require_str(cfg.get("subdir", ""), key="cif_fob.input.subdir"),
            filename=require_str(cfg.get("filename"), key="cif_fob.input.filename"),
            format=require_str(cfg.get("format", "parquet"), key="cif_fob.input.format").strip().lower(),
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.filename


@dataclass(frozen=True)
class CifFobOutputConfig:
    stage: str
    subdir: str
    format: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobOutputConfig":
        return cls(
            stage=require_str(cfg.get("stage"), key="cif_fob.output.stage").strip().lower(),
            subdir=require_str(cfg.get("subdir", ""), key="cif_fob.output.subdir"),
            format=require_str(cfg.get("format", "parquet"), key="cif_fob.output.format").strip().lower(),
        )

    def resolve_dir(self) -> Path:
        base = get_data_dir(self.stage)
        return base / self.subdir if self.subdir else base


@dataclass(frozen=True)
class CifFobSupportedFlowsConfig:
    import_code: str
    export_code: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobSupportedFlowsConfig":
        return cls(
            import_code=require_str(
                cfg.get("import"),
                key="cif_fob.source_data.supported_flows.import",
            ),
            export_code=require_str(
                cfg.get("export"),
                key="cif_fob.source_data.supported_flows.export",
            ),
        )


@dataclass(frozen=True)
class CifFobSourceColumnsConfig:
    year: str
    reporter: str
    partner: str
    flow: str
    commodity: str
    is_reported: str
    is_aggregate: str
    canonical_value: str
    source_value: str
    direct_cif_value: str
    direct_fob_value: str
    allocation_weight: str
    quantity: str
    quantity_unit: str
    quantity_estimated: str

    alt_quantity: str
    alt_quantity_unit: str
    alt_quantity_estimated: str

    net_weight: str
    net_weight_estimated: str

    gross_weight: str
    gross_weight_estimated: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobSourceColumnsConfig":
        return cls(
            year=require_str(cfg.get("year"), key="cif_fob.source_data.columns.year"),
            reporter=require_str(cfg.get("reporter"), key="cif_fob.source_data.columns.reporter"),
            partner=require_str(cfg.get("partner"), key="cif_fob.source_data.columns.partner"),
            flow=require_str(cfg.get("flow"), key="cif_fob.source_data.columns.flow"),
            commodity=require_str(cfg.get("commodity"), key="cif_fob.source_data.columns.commodity"),
            is_reported=require_str(cfg.get("is_reported"), key="cif_fob.source_data.columns.is_reported"),
            is_aggregate=require_str(cfg.get("is_aggregate"), key="cif_fob.source_data.columns.is_aggregate"),
            canonical_value=require_str(
                cfg.get("canonical_value"),
                key="cif_fob.source_data.columns.canonical_value",
            ),
            source_value=require_str(
                cfg.get("source_value"),
                key="cif_fob.source_data.columns.source_value",
            ),
            direct_cif_value=require_str(
                cfg.get("direct_cif_value"),
                key="cif_fob.source_data.columns.direct_cif_value",
            ),
            direct_fob_value=require_str(
                cfg.get("direct_fob_value"),
                key="cif_fob.source_data.columns.direct_fob_value",
            ),
            allocation_weight=require_str(
                cfg.get("allocation_weight"),
                key="cif_fob.source_data.columns.allocation_weight",
            ),
            quantity=require_str(cfg.get("quantity"), key="cif_fob.source_data.columns.quantity"),
            quantity_unit=require_str(cfg.get("quantity_unit"), key="cif_fob.source_data.columns.quantity_unit"),
            quantity_estimated=require_str(cfg.get("quantity_estimated"), key="cif_fob.source_data.columns.quantity_estimated"),

            alt_quantity=require_str(cfg.get("alt_quantity"), key="cif_fob.source_data.columns.alt_quantity"),
            alt_quantity_unit=require_str(cfg.get("alt_quantity_unit"), key="cif_fob.source_data.columns.alt_quantity_unit"),
            alt_quantity_estimated=require_str(cfg.get("alt_quantity_estimated"), key="cif_fob.source_data.columns.alt_quantity_estimated"),

            net_weight=require_str(cfg.get("net_weight"), key="cif_fob.source_data.columns.net_weight"),
            net_weight_estimated=require_str(cfg.get("net_weight_estimated"), key="cif_fob.source_data.columns.net_weight_estimated"),

            gross_weight=require_str(cfg.get("gross_weight"), key="cif_fob.source_data.columns.gross_weight"),
            gross_weight_estimated=require_str(cfg.get("gross_weight_estimated"), key="cif_fob.source_data.columns.gross_weight_estimated"),
        )


@dataclass(frozen=True)
class CifFobSourceDataConfig:
    supported_flows: CifFobSupportedFlowsConfig
    columns: CifFobSourceColumnsConfig
    allocate_supplementary_measures: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobSourceDataConfig":
        return cls(
            supported_flows=CifFobSupportedFlowsConfig.from_dict(
                require_dict(
                    cfg.get("supported_flows"),
                    key="cif_fob.source_data.supported_flows",
                )
            ),
            columns=CifFobSourceColumnsConfig.from_dict(
                require_dict(cfg.get("columns"), key="cif_fob.source_data.columns")
            ),
            allocate_supplementary_measures=require_bool(
                cfg.get("allocate_supplementary_measures", True),
                key="cif_fob.source_data.allocate_supplementary_measures",
            ),
        )


@dataclass(frozen=True)
class CifFobProductConfig:
    enable: bool
    filename: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "CifFobProductConfig":
        return cls(
            enable=require_bool(cfg.get("enable", True), key=f"{key}.enable"),
            filename=require_str(cfg.get("filename"), key=f"{key}.filename"),
        )


@dataclass(frozen=True)
class CifFobProductsConfig:
    prepared_input: CifFobProductConfig
    mirror_pairs: CifFobProductConfig
    training_sample: CifFobProductConfig
    estimated_rates: CifFobProductConfig
    corrected_trade: CifFobProductConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobProductsConfig":
        return cls(
            prepared_input=CifFobProductConfig.from_dict(
                require_dict(cfg.get("prepared_input"), key="cif_fob.products.prepared_input"),
                key="cif_fob.products.prepared_input",
            ),
            mirror_pairs=CifFobProductConfig.from_dict(
                require_dict(cfg.get("mirror_pairs"), key="cif_fob.products.mirror_pairs"),
                key="cif_fob.products.mirror_pairs",
            ),
            training_sample=CifFobProductConfig.from_dict(
                require_dict(cfg.get("training_sample"), key="cif_fob.products.training_sample"),
                key="cif_fob.products.training_sample",
            ),
            estimated_rates=CifFobProductConfig.from_dict(
                require_dict(cfg.get("estimated_rates"), key="cif_fob.products.estimated_rates"),
                key="cif_fob.products.estimated_rates",
            ),
            corrected_trade=CifFobProductConfig.from_dict(
                require_dict(cfg.get("corrected_trade"), key="cif_fob.products.corrected_trade"),
                key="cif_fob.products.corrected_trade",
            ),
        )


@dataclass(frozen=True)
class CifFobTrainingConfig:
    require_both_mirrors: bool
    require_positive_value: bool
    require_positive_quantity: bool
    minimum_quantity_concordance: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobTrainingConfig":
        return cls(
            require_both_mirrors=require_bool(
                cfg.get("require_both_mirrors", True),
                key="cif_fob.method.training.require_both_mirrors",
            ),
            require_positive_value=require_bool(
                cfg.get("require_positive_value", True),
                key="cif_fob.method.training.require_positive_value",
            ),
            require_positive_quantity=require_bool(
                cfg.get("require_positive_quantity", True),
                key="cif_fob.method.training.require_positive_quantity",
            ),
            minimum_quantity_concordance=require_float(
                cfg.get("minimum_quantity_concordance", 0.9),
                key="cif_fob.method.training.minimum_quantity_concordance",
                min_value=0.0,
                max_value=1.0,
                    ),
        )


@dataclass(frozen=True)
class CifFobOutliersConfig:
    method: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobOutliersConfig":
        return cls(
            method=require_str(cfg.get("method"), key="cif_fob.method.outliers.method"),
        )


@dataclass(frozen=True)
class CifFobMethodConfig:
    name: str
    flow_keys: list[str]
    training: CifFobTrainingConfig
    predictors: list[str]
    outliers: CifFobOutliersConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobMethodConfig":
        flow_keys_raw = cfg.get("flow_keys", [])
        predictors_raw = cfg.get("predictors", [])

        if not isinstance(flow_keys_raw, list):
            raise TypeError("cif_fob.method.flow_keys must be a list")
        if not isinstance(predictors_raw, list):
            raise TypeError("cif_fob.method.predictors must be a list")

        return cls(
            name=require_str(cfg.get("name"), key="cif_fob.method.name").strip(),
            flow_keys=[require_str(item, key="cif_fob.method.flow_keys") for item in flow_keys_raw],
            training=CifFobTrainingConfig.from_dict(
                require_dict(cfg.get("training", {}), key="cif_fob.method.training")
            ),
            predictors=[require_str(item, key="cif_fob.method.predictors") for item in predictors_raw],
            outliers=CifFobOutliersConfig.from_dict(
                require_dict(cfg.get("outliers", {}), key="cif_fob.method.outliers")
            ),
        )


@dataclass(frozen=True)
class CifFobConfig:
    input: CifFobInputConfig
    output: CifFobOutputConfig
    source_data: CifFobSourceDataConfig
    products: CifFobProductsConfig
    method: CifFobMethodConfig
    scope: PipelineScopeConfig
    engine: PipelineEngineConfig

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | None = None,
        *,
        scope: PipelineScopeConfig | None = None,
        engine: PipelineEngineConfig | None = None,
    ) -> "CifFobConfig":
        return cls.from_dict(
            load_cif_fob_config(config_path),
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
    ) -> "CifFobConfig":
        if scope is None:
            raise ValueError("CifFobConfig requires pipeline scope.")
        if engine is None:
            raise ValueError("CifFobConfig requires pipeline engine.")

        cif_fob_raw = require_dict(raw.get("cif_fob"), key="cif_fob")
        rendered = require_dict(
            _render_value(cif_fob_raw, target_hs=scope.target_hs),
            key="cif_fob",
        )

        return cls(
            input=CifFobInputConfig.from_dict(
                require_dict(rendered.get("input"), key="cif_fob.input")
            ),
            output=CifFobOutputConfig.from_dict(
                require_dict(rendered.get("output"), key="cif_fob.output")
            ),
            source_data=CifFobSourceDataConfig.from_dict(
                require_dict(rendered.get("source_data"), key="cif_fob.source_data")
            ),
            products=CifFobProductsConfig.from_dict(
                require_dict(rendered.get("products"), key="cif_fob.products")
            ),
            method=CifFobMethodConfig.from_dict(
                require_dict(rendered.get("method"), key="cif_fob.method")
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
            "mirror_pairs": self.products.mirror_pairs,
            "training_sample": self.products.training_sample,
            "estimated_rates": self.products.estimated_rates,
            "corrected_trade": self.products.corrected_trade,
        }

        if product_name not in products:
            raise KeyError(f"Unknown CIF/FOB product: {product_name!r}")

        return self.output_dir() / products[product_name].filename

    def resolved_product_paths(self) -> dict[str, Path]:
        return {
            "prepared_input": self.product_path("prepared_input"),
            "mirror_pairs": self.product_path("mirror_pairs"),
            "training_sample": self.product_path("training_sample"),
            "estimated_rates": self.product_path("estimated_rates"),
            "corrected_trade": self.product_path("corrected_trade"),
        }

    def resolved_enabled_product_paths(self) -> dict[str, Path]:
        products = {
            "prepared_input": self.products.prepared_input,
            "mirror_pairs": self.products.mirror_pairs,
            "training_sample": self.products.training_sample,
            "estimated_rates": self.products.estimated_rates,
            "corrected_trade": self.products.corrected_trade,
        }

        return {
            name: self.output_dir() / product.filename
            for name, product in products.items()
            if product.enable
        }
