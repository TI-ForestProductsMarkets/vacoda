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
    require_int,
    require_str,
)
from vacoda.config.paths import get_data_dir, get_output_dir, get_repo_references_dir
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
class CifFobLoggingConfig:
    level: str
    subdir: str
    file: str
    timestamp: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobLoggingConfig":
        return cls(
            level=require_str(
                cfg.get("level", "INFO"),
                key="cif_fob.logging.level",
            ).strip().upper(),
            subdir=require_str(
                cfg.get("subdir", "cif_fob"),
                key="cif_fob.logging.subdir",
                allow_empty=True,
            ),
            file=require_str(
                cfg.get("file", "LOG_CIF_FOB.log"),
                key="cif_fob.logging.file",
            ),
            timestamp=require_bool(
                cfg.get("timestamp", True),
                key="cif_fob.logging.timestamp",
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

    def resolve_log_file(self) -> Path:
        return self.resolve_log_dir() / self.file


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
    training_features: CifFobProductConfig
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
            training_features=CifFobProductConfig.from_dict(
                require_dict(cfg.get("training_features"), key="cif_fob.products.training_features"),
                key="cif_fob.products.training_features",
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
    minimum_quantity_concordance: float
    minimum_direct_value_coverage: float

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobTrainingConfig":
        return cls(
            minimum_quantity_concordance=require_float(
                cfg.get("minimum_quantity_concordance", 0.9),
                key="cif_fob.method.training.minimum_quantity_concordance",
                min_value=0.0,
                max_value=1.0,
            ),
            minimum_direct_value_coverage=require_float(
                cfg.get("minimum_direct_value_coverage", 0.9),
                key="cif_fob.method.training.minimum_direct_value_coverage",
                min_value=0.0,
                max_value=1.0,
            ),
        )


def _require_str_list(value: Any, *, key: str) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return [require_str(item, key=key) for item in value]


@dataclass(frozen=True)
class CifFobFeatureInputConfig:
    stage: str | None
    subdir: str
    source: str | None
    path: str | None
    filename: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, key: str) -> "CifFobFeatureInputConfig":
        filename = require_str(cfg.get("filename"), key=f"{key}.filename")

        if cfg.get("stage") is not None:
            return cls(
                stage=require_str(cfg.get("stage"), key=f"{key}.stage").strip().lower(),
                subdir=require_str(cfg.get("subdir", ""), key=f"{key}.subdir"),
                source=None,
                path=None,
                filename=filename,
            )

        if cfg.get("source") is not None:
            return cls(
                stage=None,
                subdir="",
                source=require_str(cfg.get("source"), key=f"{key}.source").strip().lower(),
                path=require_str(cfg.get("path"), key=f"{key}.path"),
                filename=filename,
            )

        raise TypeError(
            f"{key} must define either stage/subdir or source/path."
        )

    def resolve_dir(self) -> Path:
        if self.stage is not None:
            base = get_data_dir(self.stage)
            return base / self.subdir if self.subdir else base

        source = (self.source or "").strip().lower()
        if source == "repo":
            base = get_repo_references_dir()
        elif source == "data":
            base = get_data_dir("references")
        else:
            raise ValueError(
                "cif_fob feature source must be 'repo' or 'data', "
                f"got {self.source!r}."
            )

        path = Path(self.path or "")
        if path.parts and path.parts[0] == "references":
            path = Path(*path.parts[1:])

        return base / path if path.parts else base

    def resolve_path(self) -> Path:
        return self.resolve_dir() / self.filename


@dataclass(frozen=True)
class CifFobGeographyCrosswalkConfig:
    input: CifFobFeatureInputConfig
    allow_proxy_mappings: bool

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        key: str,
    ) -> "CifFobGeographyCrosswalkConfig":
        return cls(
            input=CifFobFeatureInputConfig.from_dict(
                cfg,
                key=key,
            ),
            allow_proxy_mappings=require_bool(
                cfg.get("allow_proxy_mappings", False),
                key=f"{key}.allow_proxy_mappings",
            ),
        )

    def resolve_path(self) -> Path:
        return self.input.resolve_path()


@dataclass(frozen=True)
class CifFobProductMedianUnitValueConfig:
    method: str
    minimum_observations: int
    fallback: list[str]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobProductMedianUnitValueConfig":
        return cls(
            method=require_str(
                cfg.get("method"),
                key="cif_fob.features.product.median_unit_value.method",
            ),
            minimum_observations=require_int(
                cfg.get("minimum_observations", 20),
                key="cif_fob.features.product.median_unit_value.minimum_observations",
                min_value=1,
            ),
            fallback=_require_str_list(
                cfg.get("fallback", []),
                key="cif_fob.features.product.median_unit_value.fallback",
            ),
        )


@dataclass(frozen=True)
class CifFobProductFeaturesConfig:
    group_digits: int
    median_unit_value: CifFobProductMedianUnitValueConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobProductFeaturesConfig":
        return cls(
            group_digits=require_int(
                cfg.get("group_digits", 4),
                key="cif_fob.features.product.group_digits",
                min_value=1,
            ),
            median_unit_value=CifFobProductMedianUnitValueConfig.from_dict(
                require_dict(
                    cfg.get("median_unit_value", {}),
                    key="cif_fob.features.product.median_unit_value",
                )
            ),
        )


@dataclass(frozen=True)
class CifFobFeaturesConfig:
    geography: CifFobFeatureInputConfig
    pair_geography: CifFobFeatureInputConfig
    geography_crosswalk: CifFobGeographyCrosswalkConfig
    oil_price: CifFobFeatureInputConfig
    product: CifFobProductFeaturesConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobFeaturesConfig":
        return cls(
            geography=CifFobFeatureInputConfig.from_dict(
                require_dict(
                    cfg.get("geography"),
                    key="cif_fob.features.geography",
                ),
                key="cif_fob.features.geography",
            ),
            pair_geography=CifFobFeatureInputConfig.from_dict(
                require_dict(
                    cfg.get("pair_geography"),
                    key="cif_fob.features.pair_geography",
                ),
                key="cif_fob.features.pair_geography",
            ),
            geography_crosswalk=CifFobGeographyCrosswalkConfig.from_dict(
                require_dict(
                    cfg.get("geography_crosswalk"),
                    key="cif_fob.features.geography_crosswalk",
                ),
                key="cif_fob.features.geography_crosswalk",
            ),
            oil_price=CifFobFeatureInputConfig.from_dict(
                require_dict(
                    cfg.get("oil_price"),
                    key="cif_fob.features.oil_price",
                ),
                key="cif_fob.features.oil_price",
            ),
            product=CifFobProductFeaturesConfig.from_dict(
                require_dict(
                    cfg.get("product"),
                    key="cif_fob.features.product",
                )
            ),
        )



@dataclass(frozen=True)
class CifFobEstimationSourceWeightsConfig:
    direct_reported_weight: float
    direct_estimated_weight: float
    mirror_reported_weight: float
    mirror_estimated_weight: float

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobEstimationSourceWeightsConfig":
        return cls(
            direct_reported_weight=require_float(
                cfg.get("direct_reported_weight", 1.0),
                key=(
                    "cif_fob.method.estimation.source_weights."
                    "direct_reported_weight"
                ),
                min_value=0.0,
            ),
            direct_estimated_weight=require_float(
                cfg.get("direct_estimated_weight", 0.75),
                key=(
                    "cif_fob.method.estimation.source_weights."
                    "direct_estimated_weight"
                ),
                min_value=0.0,
            ),
            mirror_reported_weight=require_float(
                cfg.get("mirror_reported_weight", 0.5),
                key=(
                    "cif_fob.method.estimation.source_weights."
                    "mirror_reported_weight"
                ),
                min_value=0.0,
            ),
            mirror_estimated_weight=require_float(
                cfg.get("mirror_estimated_weight", 0.25),
                key=(
                    "cif_fob.method.estimation.source_weights."
                    "mirror_estimated_weight"
                ),
                min_value=0.0,
            ),
        )


@dataclass(frozen=True)
class CifFobEstimationValidationConfig:
    strategy: str
    holdout_year: int | None

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobEstimationValidationConfig":
        strategy = require_str(
            cfg.get("strategy", "latest_year"),
            key="cif_fob.method.estimation.validation.strategy",
        ).strip().lower()

        holdout_year_raw = cfg.get("holdout_year")

        return cls(
            strategy=strategy,
            holdout_year=(
                require_int(
                    holdout_year_raw,
                    key=(
                        "cif_fob.method.estimation.validation."
                        "holdout_year"
                    ),
                )
                if holdout_year_raw is not None
                else None
            ),
        )


@dataclass(frozen=True)
class CifFobEstimationModelConfig:
    name: str
    loss: str
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_bins: int
    random_state: int

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobEstimationModelConfig":
        max_bins = require_int(
            cfg.get("max_bins", 255),
            key="cif_fob.method.estimation.model.max_bins",
            min_value=2,
        )
        if max_bins > 255:
            raise ValueError(
                "'cif_fob.method.estimation.model.max_bins' must be <= 255, "
                f"got {max_bins}."
            )
        return cls(
            name=require_str(
                cfg.get("name", "hist_gradient_boosting"),
                key="cif_fob.method.estimation.model.name",
            ).strip().lower(),
            loss=require_str(
                cfg.get("loss", "absolute_error"),
                key="cif_fob.method.estimation.model.loss",
            ).strip().lower(),
            learning_rate=require_float(
                cfg.get("learning_rate", 0.05),
                key="cif_fob.method.estimation.model.learning_rate",
                min_value=0.0,
            ),
            max_iter=require_int(
                cfg.get("max_iter", 300),
                key="cif_fob.method.estimation.model.max_iter",
                min_value=1,
            ),
            max_leaf_nodes=require_int(
                cfg.get("max_leaf_nodes", 31),
                key=(
                    "cif_fob.method.estimation.model."
                    "max_leaf_nodes"
                ),
                min_value=2,
            ),
            min_samples_leaf=require_int(
                cfg.get("min_samples_leaf", 50),
                key=(
                    "cif_fob.method.estimation.model."
                    "min_samples_leaf"
                ),
                min_value=1,
            ),
            l2_regularization=require_float(
                cfg.get("l2_regularization", 1.0),
                key=(
                    "cif_fob.method.estimation.model."
                    "l2_regularization"
                ),
                min_value=0.0,
            ),
            max_bins=max_bins,
            random_state=require_int(
                cfg.get("random_state", 42),
                key="cif_fob.method.estimation.model.random_state",
            ),
        )


@dataclass(frozen=True)
class CifFobEstimationConfig:
    minimum_evidence_value_usd: float
    minimum_factor: float
    maximum_factor: float
    exclude_proxy_weights: bool
    training_factor_sources: list[str]
    categorical_predictors: list[str]
    source_weights: CifFobEstimationSourceWeightsConfig
    validation: CifFobEstimationValidationConfig
    model: CifFobEstimationModelConfig

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "CifFobEstimationConfig":
        minimum_factor = require_float(
            cfg.get("minimum_factor", 1.0),
            key="cif_fob.method.estimation.minimum_factor",
            min_value=0.0,
        )

        maximum_factor = require_float(
            cfg.get("maximum_factor", 3.0),
            key="cif_fob.method.estimation.maximum_factor",
            min_value=minimum_factor,
        )

        training_factor_sources = _require_str_list(
                                    cfg.get(
                                        "training_factor_sources", 
                                        ['direct','mirror']
                                        ), 
                                    key=("cif_fob.method.estimation."
                                         "training_factor_sources"
                                         ),
                )
        supported_factor_sources = {"direct", "mirror"} 
        unsupported_factor_sources = sorted( 
                                set(training_factor_sources) - 
                                supported_factor_sources ) 
        if unsupported_factor_sources: 
            raise ValueError( "Unsupported CIF/FOB training factor sources: " + ", ".join(unsupported_factor_sources) ) 
        if not training_factor_sources: 
            raise ValueError( "cif_fob.method.estimation.training_factor_sources " "must contain at least one source." )

        categorical_predictors = _require_str_list(
            cfg.get("categorical_predictors", []),
            key=(
                "cif_fob.method.estimation."
                "categorical_predictors"
            ),
        )

        return cls(
            minimum_evidence_value_usd=require_float(
                cfg.get("minimum_evidence_value_usd", 1000.0),
                key=(
                    "cif_fob.method.estimation."
                    "minimum_evidence_value_usd"
                ),
                min_value=0.0,
            ),
            minimum_factor=minimum_factor,
            maximum_factor=maximum_factor,
            exclude_proxy_weights=require_bool(
                cfg.get("exclude_proxy_weights", True),
                key=(
                    "cif_fob.method.estimation."
                    "exclude_proxy_weights"
                ),
            ),
            training_factor_sources=training_factor_sources,
            categorical_predictors=categorical_predictors,
            source_weights=(
                CifFobEstimationSourceWeightsConfig.from_dict(
                    require_dict(
                        cfg.get("source_weights", {}),
                        key=(
                            "cif_fob.method.estimation."
                            "source_weights"
                        ),
                    )
                )
            ),
            validation=(
                CifFobEstimationValidationConfig.from_dict(
                    require_dict(
                        cfg.get("validation", {}),
                        key=(
                            "cif_fob.method.estimation."
                            "validation"
                        ),
                    )
                )
            ),
            model=CifFobEstimationModelConfig.from_dict(
                require_dict(
                    cfg.get("model", {}),
                    key="cif_fob.method.estimation.model",
                )
            ),
        )


@dataclass(frozen=True)
class CifFobPredictorsConfig:
    bilateral: list[str]
    fuel_and_time: list[str]
    product: list[str]
    route: list[str]
    optional_connectivity: list[str]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobPredictorsConfig":
        return cls(
            bilateral=_require_str_list(
                cfg.get("bilateral", []),
                key="cif_fob.method.predictors.bilateral",
            ),
            fuel_and_time=_require_str_list(
                cfg.get("fuel_and_time", []),
                key="cif_fob.method.predictors.fuel_and_time",
            ),
            product=_require_str_list(
                cfg.get("product", []),
                key="cif_fob.method.predictors.product",
            ),
            route=_require_str_list(
                cfg.get("route", []),
                key="cif_fob.method.predictors.route",
            ),
            optional_connectivity=_require_str_list(
                cfg.get("optional_connectivity", []),
                key="cif_fob.method.predictors.optional_connectivity",
            ),
        )

    def all(self) -> list[str]:
        return (
            self.bilateral
            + self.fuel_and_time
            + self.product
            + self.route
            + self.optional_connectivity
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
    flow_keys: list[str]
    training: CifFobTrainingConfig
    estimation: CifFobEstimationConfig
    predictors: CifFobPredictorsConfig
    outliers: CifFobOutliersConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CifFobMethodConfig":
        flow_keys_raw = cfg.get("flow_keys", [])
        predictors_raw = cfg.get("predictors", {})

        if not isinstance(flow_keys_raw, list):
            raise TypeError("cif_fob.method.flow_keys must be a list")
        if not isinstance(predictors_raw, dict):
            raise TypeError("cif_fob.method.predictors must be a dictionary")

        return cls(
            flow_keys=[require_str(item, key="cif_fob.method.flow_keys") for item in flow_keys_raw],
            training=CifFobTrainingConfig.from_dict(
                require_dict(cfg.get("training", {}), key="cif_fob.method.training")
            ),
            estimation=CifFobEstimationConfig.from_dict(
                require_dict(cfg.get("estimation", {}), key="cif_fob.method.estimation")
            ),
            predictors=CifFobPredictorsConfig.from_dict(
                require_dict(cfg.get("predictors", {}), key="cif_fob.method.predictors")
            ),
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
    logging: CifFobLoggingConfig
    features: CifFobFeaturesConfig
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
            logging=CifFobLoggingConfig.from_dict(
                require_dict(rendered.get("logging", {}), key="cif_fob.logging")
            ),
            features=CifFobFeaturesConfig.from_dict(
                require_dict(rendered.get("features"), key="cif_fob.features")
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
            "training_features": self.products.training_features,
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
            "training_features": self.product_path("training_features"),
            "estimated_rates": self.product_path("estimated_rates"),
            "corrected_trade": self.product_path("corrected_trade"),
        }

    def resolved_enabled_product_paths(self) -> dict[str, Path]:
        products = {
            "prepared_input": self.products.prepared_input,
            "mirror_pairs": self.products.mirror_pairs,
            "training_sample": self.products.training_sample,
            "training_features": self.products.training_features,
            "estimated_rates": self.products.estimated_rates,
            "corrected_trade": self.products.corrected_trade,
        }

        return {
            name: self.output_dir() / product.filename
            for name, product in products.items()
            if product.enable
        }
